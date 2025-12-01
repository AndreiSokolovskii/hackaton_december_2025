import torch
from enum import Enum
from contextlib import contextmanager
from einops import rearrange 
from Bio.Data import IUPACData
import numpy as np 
from torch import Tensor
aa_to_idx = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
        # add noncanonicals 
        "DM0": 11, 
        "MLY": 11, 
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "XAA":20,
}

idx_to_aa = {v: k for k, v in aa_to_idx.items()}
idx_to_aa.update({11: "LYS"})

idx_to_aa1 = {k: IUPACData.protein_letters_3to1_extended[v.capitalize()] for k, v in idx_to_aa.items()}
aa1_to_idx = {v: k for k, v in idx_to_aa1.items()}
min_norm_clamp = 1e-9
cos_max, cos_min = (1 - 1e-9), -(1 - 1e-9)
def S_to_seq(S):return "".join([IUPACData.protein_letters_3to1_extended[idx_to_aa[token].capitalize()] for token in S.tolist()])
def _dihedrals(X, eps=1e-7):
        X = torch.reshape(X, [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = div_norm(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]
        n_2 = div_norm(torch.cross(u_2, u_1), dim=-1)
        n_1 = div_norm(torch.cross(u_1, u_0), dim=-1)
        
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        D = torch.nn.functional.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])

        return torch.cat([torch.cos(D), torch.sin(D)], 1)
def div_norm(tensor, dim=-1): return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))
def _orientations(X):
    forward = div_norm(X[1:] - X[:-1])
    backward = div_norm(X[:-1] - X[1:])
    forward = torch.nn.functional.pad(forward, [0, 0, 0, 1])
    backward = torch.nn.functional.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def _sidechains(n, origin, c):
    c, n = div_norm(c - origin), div_norm(n - origin)
    bisector = div_norm(c + n)
    perp = div_norm(torch.cross(c, n))
    vec = -bisector * 3**(-0.5) - perp * (2 / 3)**0.5
    return vec 

def safe_norm(x, dim, keepdim = False, eps = 1e-12): return torch.sqrt(torch.sum(torch.square(x), dim=dim, keepdim=keepdim) + eps)
def safe_normalize(x, eps = 1e-12, dim=-1): return x / safe_norm(x, dim=dim, keepdim=True, eps=eps)
def rotation_from_3_points(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    e1 = safe_normalize(v1)
    e2 = safe_normalize(v2 - (torch.sum(e1 * v2, dim=-1, keepdim=True) * e1))
    e3 = safe_normalize(torch.cross(e1, e2))
    rot = torch.cat((e1, e2, e3), dim=-1).reshape(-1, 3, 3)
    return rearrange(rot, "... i j->... j i")

def torch_norm(x:Tensor, axis:int=-1, keepdims:bool=True, eps:float=1e-8)->Tensor: return torch.sqrt(torch.sum(x ** 2, dim=axis, keepdim=keepdims) + eps)
def normalize(v:Tensor, eps:float=1e-8) -> Tensor: return v / torch_norm(v, eps=eps)

def get_o_from_atoms(a:Tensor, b:Tensor, c:Tensor, L:float=1.231,A:float=2.108, D:float=-3.142, eps:float=1e-8)->Tensor:    
    bc = normalize(b - c)
    n = normalize(torch.cross(b - a, bc, dim=-1))
    return c + L * (np.cos(A) * bc + np.sin(A) * np.cos(D) * torch.cross(n, bc, dim=-1) + np.sin(A) * np.sin(D) * (-n))

EPS = 1e-8

def _norm(v, dim=-1, keepdim=True):
    return v / (v.norm(dim=dim, keepdim=keepdim).clamp_min(EPS))


def get_4th_atom(
    a_coord: torch.Tensor,  
    b_coord: torch.Tensor,
    c_coord: torch.Tensor,
    length: float,
    planar: float,
    dihedral: float,
) -> torch.Tensor:
    device = a_coord.device
    dtype = a_coord.dtype
    length = torch.tensor(length, device=device, dtype=dtype)
    planar = torch.tensor(planar, device=device, dtype=dtype)
    dihedral = torch.tensor(dihedral, device=device, dtype=dtype)
    
    # normalize bc vector = (B - C)
    bc_vec = b_coord - c_coord
    bc_vec = _norm(bc_vec, dim=-1, keepdim=True)  # shape (...,3)

    # vector for computing normal: (B - A)
    ba_vec = b_coord - a_coord
    ba_vec = _norm(ba_vec, dim=-1, keepdim=True)

    # normal to plane
    n_vec = torch.cross(ba_vec, bc_vec, dim=-1)
    n_vec = _norm(n_vec, dim=-1, keepdim=True)

    # second in-plane perpendicular vector
    t_vec = torch.cross(n_vec, bc_vec, dim=-1)  # already orthogonal
    cos_p = torch.cos(planar)
    sin_p = torch.sin(planar)
    cos_d = torch.cos(dihedral)
    sin_d = torch.sin(dihedral)

    d0 = length * cos_p
    d1 = length * sin_p * cos_d
    d2 = -length * sin_p * sin_d  # minus sign per your original formula

    for d in (d0, d1, d2):
        if d.dim() == 0:
            d = d.unsqueeze(0)
    d_coord = c_coord + d0.unsqueeze(-1) * bc_vec + d1.unsqueeze(-1) * t_vec + d2.unsqueeze(-1) * n_vec

    d_coord = torch.nan_to_num(d_coord, nan=0.0, posinf=0.0, neginf=0.0)

    return d_coord


def place_missing_cb_v2(N, CA, C):
    length = 1.522
    planar = 1.927   # ~110.1 deg in rad
    dihedral = -2.143
    cb_coords = get_4th_atom(C, N, CA, length, planar, dihedral)
    return cb_coords

@contextmanager
def disable_tf32():
    """temporarily disable 32-bit float ops"""
    if torch.cuda.is_available():
        orig_value = torch.backends.cuda.matmul.allow_tf32  # noqa
        torch.backends.cuda.matmul.allow_tf32 = False  # noqa
        yield
        torch.backends.cuda.matmul.allow_tf32 = orig_value  # noqa
    else:
        yield
class TrRosettaOrientationType(Enum):
    PHI = ['N', 'CA', 'CB', 'CB']
    PSI = ['CA', 'CB', 'CB']
    OMEGA = ['CA', 'CB', 'CB', 'CA']

def unsigned_angle_all(ps):
    
    device_type = "cpu" if ps[0].device.type == "cpu" else "cuda"
    with disable_tf32(), torch.autocast(device_type=device_type, enabled=False):
        p0, p1, p2 = ps[0], ps[1], ps[2]
        b01, b12 = p0 - p1, p2.unsqueeze(-3) - p1.unsqueeze(-2)
        M = b01.unsqueeze(-2) * b12
        n01, n12 = torch.norm(b01, dim=-1, keepdim=True), torch.norm(b12, dim=-1)
        prods = torch.clamp_min(n01 * n12, min_norm_clamp)
        cos_theta = torch.sum(M, dim=-1) / prods
        cos_theta[cos_theta < cos_min] = cos_min
        cos_theta[cos_theta > cos_max] = cos_max
    return torch.acos(cos_theta)
def signed_dihedral_all_12(ps):
    device_type = "cpu" if ps[0].device.type == "cpu" else "cuda"
    with disable_tf32(), torch.autocast(device_type=device_type, enabled=False):
        p0, p1, p2, p3 = ps
        b0, b1, b2 = p0 - p1, p2.unsqueeze(-3) - p1.unsqueeze(-2), p3 - p2
        b1 = b1 / torch.norm(b1, dim=-1, keepdim=True).clamp_min(min_norm_clamp)
        v = b0.unsqueeze(-2) - torch.sum(b0.unsqueeze(-2) * b1, dim=-1, keepdim=True) * b1
        w = b2.unsqueeze(-3) - torch.sum(b2.unsqueeze(-3) * b1, dim=-1, keepdim=True) * b1
        x = torch.sum(v * w, dim=-1)
        y = torch.sum(torch.cross(b1, v) * w, dim=-1)
    return torch.atan2(y, x)

def signed_dihedral_all_123(ps):
    device_type = "cpu" if ps[0].device.type == "cpu" else "cuda"
    with disable_tf32(), torch.autocast(device_type=device_type, enabled=False):
        p0, p1, p2, p3 = ps
        b0, b1, b2 = p0 - p1, p2 - p1, p3.unsqueeze(-3) - p2.unsqueeze(-2)
        b1 = b1 / torch.norm(b1, dim=-1, keepdim=True).clamp_min(min_norm_clamp)
        v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
        w = b2 - torch.sum(b2 * b1.unsqueeze(-2), dim=-1, keepdim=True) * b1.unsqueeze(-2)
        x = torch.sum(v.unsqueeze(-2) * w, dim=-1)
        y = torch.sum(torch.cross(b1, v).unsqueeze(-2) * w, dim=-1)
        ret = torch.atan2(y, x)
    return ret
def get_tr_rosetta_orientation_mat(N, CA, CB, ori_type):

    if ori_type == TrRosettaOrientationType.PSI:
        mat = unsigned_angle_all([CA, CB, CB])
    elif ori_type == TrRosettaOrientationType.OMEGA:
        mat = signed_dihedral_all_12([CA, CB, CB, CA])
    elif ori_type == TrRosettaOrientationType.PHI:
        mat = signed_dihedral_all_123([N, CA, CB, CB])
    else:
        raise Exception(f'dihedral type {ori_type} not accepted')
    # expand back to full size
    return mat

def get_tr_rosetta_orientation_mats(N, CA, CB):
    phi = get_tr_rosetta_orientation_mat(N, CA, CB, TrRosettaOrientationType.PHI)
    psi = get_tr_rosetta_orientation_mat(N, CA, CB, TrRosettaOrientationType.PSI)
    omega = get_tr_rosetta_orientation_mat(N, CA, CB, TrRosettaOrientationType.OMEGA)
    return phi, psi, omega

def tr_rosetta_edge_attr(N, Ca, Cb, data_edge_index):
    phi, psi, omega = get_tr_rosetta_orientation_mats(N=N, CA=Ca, CB=Cb)
    return torch.stack([phi[data_edge_index[0], data_edge_index[1]], psi[data_edge_index[0], data_edge_index[1]], omega[data_edge_index[0], data_edge_index[1]], \
        phi[data_edge_index[1], data_edge_index[0]], psi[data_edge_index[1], data_edge_index[0]], omega[data_edge_index[1], data_edge_index[0]]], dim=-1)

def rbf_encode(distances, num_rbf=16, min_distance=2.0, max_distance=32.0):
    rbf_centers = torch.linspace(min_distance, max_distance, num_rbf, device=distances.device)
    rbf_widths = (max_distance / num_rbf) * torch.ones_like(rbf_centers)
    rbf_activation = torch.exp(-((distances.unsqueeze(-1) - rbf_centers) ** 2) / (2 * rbf_widths**2))
    return rbf_activation

def fourier_encode_dist(x, num_encodings = 4, include_self = False):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x