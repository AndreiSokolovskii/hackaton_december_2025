import warnings
from torch.distributions import Categorical
import torch_geometric
import torch
from Bio.PDB import PDBParser
import argparse
import copy
import numpy as np
import glob
import utils
from utils import S_to_seq
from itertools import product
from torch_geometric.nn import radius_graph
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE, Compose
from torch_geometric.data import Data as PyG_data
import os.path as osp
import pydssp
from torch_geometric.data import Batch
try:
    get_ipython
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
def fill_label_mask(
    spec: str = '0',
    length: int = None,
    device: str='cpu'
) -> torch.Tensor:
    """
    Create a 1D boolean tensor filled with False, setting True at positions specified by spec.
    Supports:
      - spec = 0: all False tensor
      - single indices: "5"
      - inclusive ranges: "11:15" means [11..15]
      - prefix ranges: ":10" marks first 10 positions [0..9]
      - suffix ranges: "-60:" marks last 60 positions [n-60..n-1]
      - relative ranges: "-10:-5" marks positions [n-10..n-5] inclusive
      - combination of tokens separated by spaces
    If length is None, inferred as necessary to accommodate all tokens except suffix and relative ranges (which require length).

    Args:
        spec: -1 or specification string
        length: Optional tensor length
    Returns:
        torch.BoolTensor of shape (length,), True at specified indices
    Raises:
        ValueError or TypeError on invalid input or bounds
    """
    # Default: all False
    from typing import Set
    if spec == '0':
        n = length if length is not None else 0
        if n < 0:
            raise ValueError(f"Tensor length must be non-negative, got {n}")
        return torch.zeros(n, dtype=torch.bool)

    if not isinstance(spec, str):
        raise TypeError(f"spec must be a string or -1, got {type(spec)}")

    tokens = spec.split()
    # Check that suffix and relative ranges have a defined length
    if any((tok.startswith('-') and tok.endswith(':')) or (tok.count(':')==1 and tok.startswith('-') and tok.split(':')[1].startswith('-')) for tok in tokens) and length is None:
        raise ValueError("Suffix ranges like '-N:' or relative ranges '-A:-B' require a defined length")

    # Determine tensor length if not provided
    if length is not None:
        n = length
    else:
        max_end = 0
        for tok in tokens:
            if tok.startswith(':') and tok[1:].isdigit():
                end = int(tok[1:])
                max_end = max(max_end, end)
            elif ':' in tok and not tok.startswith(':') and not tok.endswith(':'):
                a_str, b_str = tok.split(':')
                if a_str.isdigit() and b_str.isdigit():
                    a, b = int(a_str), int(b_str)
                    max_end = max(max_end, max(a, b) + 1)
                # skip relative or suffix here
            elif tok.isdigit():
                idx = int(tok)
                max_end = max(max_end, idx + 1)
        n = max_end

    if n < 0:
        raise ValueError(f"Tensor length must be non-negative, got {n}")

    positions: Set[int] = set()
    for tok in tokens:
        # prefix range :N
        if tok.startswith(':') and tok[1:].isdigit():
            end = int(tok[1:])
            positions.update(range(0, end))
        # suffix range -N:
        elif tok.startswith('-') and tok.endswith(':') and tok[1:-1].isdigit():
            count = int(tok[1:-1])
            if count > n:
                raise ValueError(f"Suffix count {count} exceeds tensor length {n}")
            positions.update(range(n - count, n))
        # relative range -A:-B
        elif ':' in tok and tok.startswith('-'):
            a_str, b_str = tok.split(':')
            if a_str.startswith('-') and b_str.startswith('-') and a_str[1:].isdigit() and b_str[1:].isdigit():
                start = n + int(a_str)
                end = n + int(b_str)
                start, end = min(start, end), max(start, end)
                positions.update(range(start, end + 1))
            else:
                # inclusive a:b with possible mixed signs
                a, b = int(a_str), int(b_str)
                start, end = min(a, b), max(a, b)
                positions.update(range(start, end + 1))
        # inclusive a:b (both positive)
        elif ':' in tok:
            a_str, b_str = tok.split(':')
            a, b = int(a_str), int(b_str)
            start, end = min(a, b), max(a, b)
            positions.update(range(start, end + 1))
        # single index
        else:
            idx = int(tok)
            if idx < 0:
                idx = n + idx
            positions.add(idx)

    # Validate positions
    invalid = [pos for pos in positions if pos < 0 or pos >= n]
    if invalid:
        raise ValueError(f"Positions out of bounds for length {n}: {invalid}")

    # Build tensor
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    idx_tensor = torch.tensor(sorted(positions), dtype=torch.long, device=device)
    mask[idx_tensor] = True
    return mask.logical_not()

def parse_pdb_to_data(file, device = 'cpu', pre_clean=False):
    parser = PDBParser(QUIET=True)
    if pre_clean:
        from clean_script import main as cleaner
        import os 
        input_file = file
        filename = file.split('/')[-1].split('.')[0]
        outfile='__pycache__/' + filename+'_clean.pdb'
        cleaner(input_file, outfile=outfile)
        structure = parser.get_structure("protein", outfile)
        os.system(f'rm {outfile}')
    else:
        structure = parser.get_structure("protein", file)
    residues = list(structure.get_residues())   
    y = torch.tensor([utils.aa_to_idx[residue.resname] for residue in residues if "CA" in residue ], device=device)
    atoms_coord = {}
    for atom in ['N', 'CA', 'C']:
        atoms_coord[atom] = torch.tensor(np.array([residue[atom].coord for residue in residues if atom in residue]), device=device)
        
    fake_o_pos = np.where(np.array([0 if "O" in residue else 1 for residue in residues]))[0].tolist()
    o_real_coord= {i:residue["O"] for i, residue in enumerate(residues) if "O" in residue}
    o_fake_coord = utils.get_o_from_atoms(torch.roll(atoms_coord['N'], shifts=-1, dims=0) , atoms_coord['CA'], atoms_coord['C'])
    o_fake = {item:o_fake_coord[item] for item in fake_o_pos}
    o_real_coord.update(o_fake)
    atoms_coord['O'] = torch.tensor(np.array([residues[i]["O"].coord if "O" in residues[i] else o_fake[i] for i in range(len(residues))]), dtype=torch.float, device=device)
    return PyG_data(pos=torch.cat([atoms_coord['O'] , atoms_coord['C'], atoms_coord['CA'], atoms_coord['N']], dim=1).to(torch.float32), y=y)

def data_featurize(data, num_rbf=16, radius=9.5, max_n=48, device='cuda'):
    
    if not hasattr(data, 'label_mask'):
        data.label_mask = torch.zeros_like(data.y, dtype=torch.bool, device=data.y.device)
    if not hasattr(data, 'batch'):
        data.label_mask = torch.zeros_like(data.y, device=data.y.device)
    prot_x = data.pos.to(device)
    
    Ca = prot_x[:, 6:9]
    C  = prot_x[:, 3:6]
    O  = prot_x[:, :3]
    N  = prot_x[:, 9:]   
    atoms= torch.stack([N, Ca, C, O], dim=1)
    edge_index = radius_graph(x = Ca, r =radius, loop = True, max_num_neighbors=max_n)
    dssp = pydssp.assign(atoms, out_type='onehot').to(torch.float32)

    
            
    data.edge_index = edge_index
    tr1 = AddLaplacianEigenvectorPE(k=10)
    tr2 = AddRandomWalkPE(walk_length=10)
    tr = Compose([tr1, tr2])
    data = tr(data)
    Cb = utils.place_missing_cb_v2(N, Ca, C)
     
    x = utils._dihedrals(torch.stack([N, Ca, C], dim=1))
    x2 = utils._dihedrals(torch.stack([N, C, O], dim=1))
    data.x = torch.cat([dssp.to(device), x, x2], dim=-1)
    
    orientations = utils._orientations(Cb)
    sidechains = utils._sidechains(N, Cb, C)
    rots = utils.rotation_from_3_points(N, Cb, C)
    quats = rots.reshape(O.shape[0], 9)
    data.x = torch.cat([data.x, sidechains, quats, orientations.reshape(O.shape[0], 6)], dim=-1)

    edge_index = data.edge_index
    edge_attr = utils.tr_rosetta_edge_attr(N, Ca, Cb, edge_index)

    row, col = edge_index[0], edge_index[1]
    all_dist = []
    for A, B in list(product([N, C, O, Ca, Cb ], repeat=2)): 
        distances = utils.rbf_encode(torch.norm(A[row] - B[col], dim=1), num_rbf=num_rbf)
        f_en = utils.fourier_encode_dist(torch.square(A[row] - B[col]).sum(-1), include_self=False)
        distances = torch.cat([distances, f_en], dim=1)
        all_dist.append(distances)
    
    edge_attr2 = torch.cat(all_dist, dim=1)
    data.edge_attr = torch.cat([edge_attr, edge_attr2], dim=-1)
    return data

def parse_and_featurize_light(pdb_path,design_position='0', device='cpu', pre_clean=True, data=None):
    if  data is None:
        data = parse_pdb_to_data(pdb_path, device=device, pre_clean=pre_clean)
        data.label_mask = fill_label_mask(spec=design_position, length=data.y.shape[0], device=device)
    else:
        data = PyG_data(pos=data.pos, y=data.y, label_mask=data.label_mask)
    data = data_featurize(data, device=device)
    return data

@torch.no_grad()
def one_shot_fast(model, data, temperature, verbose, suppress = None, eol=False, num_seq_per_target=1):
    real_y = data.y.clone()
    label_mask = data.label_mask
    pos_to_design = torch.where(label_mask.logical_not())[0]
    res_y = torch.zeros_like(data.y)
    res_y[label_mask] = data.y[label_mask]
    y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y, label_mask=data.label_mask, batch=torch.zeros_like(data.y), laplacian_eigenvector_pe=data.laplacian_eigenvector_pe, random_walk_pe=data.random_walk_pe) 

    if suppress is not None: y_hat[:, suppress] = -1e9
    K = num_seq_per_target
    dist = Categorical(logits=y_hat / temperature)
    samples = dist.sample(sample_shape=(K,))
    out_Conf = []
    out_Sr = []
    out_Seq = []
    for i in range(K):
        sample = samples[i]
        res_y[label_mask.logical_not()] = sample[label_mask.logical_not()]
        res_conf = dist.log_prob(sample).exp()[label_mask.logical_not()]
        confidence = res_conf.mean().item()
        seq_recovery = res_y.eq(real_y).sum() / real_y.shape[0]
        out_Conf += [confidence]
        out_Sr += [seq_recovery.item()]
        out_Seq += [S_to_seq(res_y)]
    return out_Seq, out_Sr, out_Conf

@torch.no_grad()
def one_shot_diverse(model, data, temperature, verbose, suppress = None, eol=False):
    real_y = data.y.clone()
    label_mask = data.label_mask
    pos_to_design = torch.where(label_mask.logical_not())[0]
    res_y = torch.zeros_like(data.y)
    res_y[label_mask] = data.y[label_mask]
    y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y, label_mask=data.label_mask, batch=torch.zeros_like(data.y), laplacian_eigenvector_pe=data.laplacian_eigenvector_pe, random_walk_pe=data.random_walk_pe) 
    if suppress is not None: y_hat[:, suppress] = -1e9
    dist = Categorical(logits=y_hat / temperature)
    sample = dist.sample()
    res_y[label_mask.logical_not()] = sample[label_mask.logical_not()]
    res_conf = dist.log_prob(sample).exp()[label_mask.logical_not()]    
    confidence = res_conf.mean().item()
    seq_recovery = res_y.eq(real_y).sum() / real_y.shape[0]
    return S_to_seq(res_y), seq_recovery.item(), confidence
@torch.no_grad()
def one_shot_fast_batch(model, data, temperature, verbose, suppress = None, eol=False, num_seq_per_target=1):
    real_y = data.y.clone()
    label_mask = data.label_mask
    pos_to_design = torch.where(label_mask.logical_not())[0]
    res_y = torch.zeros_like(data.y)
    res_y[label_mask] = data.y[label_mask]
    y_hat = model(x=data.x, edge_index=data.edge_index,
                   edge_attr=data.edge_attr, y=data.y,
                     label_mask=data.label_mask, batch=data.batch,
                       laplacian_eigenvector_pe=data.laplacian_eigenvector_pe, random_walk_pe=data.random_walk_pe) 

    if suppress is not None: y_hat[:, suppress] = -1e9
    K = num_seq_per_target
    dist = Categorical(logits=y_hat / temperature)
    samples = dist.sample(sample_shape=(K,))
    out_Conf = []
    out_Sr = []
    out_Seq = []
    for i in range(K):
        sample = samples[i]
        res_y[label_mask.logical_not()] = sample[label_mask.logical_not()]
        res_conf = dist.log_prob(sample).exp()[label_mask.logical_not()]
        seq_recovery = res_y.eq(real_y).sum() / real_y.shape[0]
        out_Conf += [res_conf]
        out_Sr += [seq_recovery.item()]
        out_Seq += [S_to_seq(res_y)]
    return out_Seq, out_Sr, out_Conf

@torch.no_grad()
def iterative_shot(model, data, temperature, verbose, suppress= None, eol=False, num_seq_per_target=1):
    real_y = data.y.clone()
    label_mask = data.label_mask.clone()
    pos_to_design = torch.where(label_mask.logical_not())[0]
    pos_to_design = pos_to_design[torch.randperm(pos_to_design.shape[0])]

    res_y = torch.zeros_like(data.y)
    res_y[label_mask] = data.y[label_mask]
    res_conf = []
    
    inner_loop = tqdm(pos_to_design, desc='Residues sampling', position=2 if eol else 1, leave=False) if verbose else pos_to_design
    for i in inner_loop:
        data.y=res_y
        data.label_mask = label_mask
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y, label_mask=data.label_mask, batch=torch.zeros_like(data.y), laplacian_eigenvector_pe=data.laplacian_eigenvector_pe, random_walk_pe=data.random_walk_pe) 
        if suppress is not None: y_hat[:, suppress] = -1e9
        dist = Categorical(logits=y_hat/ temperature )
        sample = dist.sample()
        conf_sample = dist.log_prob(sample).exp() 
        res_y[i] = sample[i]
        res_conf += [conf_sample[i]]
        data.label_mask[i] = True
    
    confidence = torch.tensor(res_conf).mean().item()
    seq_recovery = res_y.eq(real_y).sum() / real_y.shape[0] 
    return S_to_seq(res_y), seq_recovery.item(), confidence
@torch.no_grad()
def warmup_model(model, device):
    num_nodes = 10
    for _ in range(5):
            num_edges = num_nodes * 20
            x = torch.randn((num_nodes, 33), device=device)
            edge_index = torch.randint(low=0, high=num_nodes, size = (2, num_edges), device=device)
            edge_attr = torch.randn((num_edges, 606), device=device)
            y = torch.randint(low=0, high=19, size = (num_nodes,), device=device)
            label_mask = torch.zeros_like(y, device=device, dtype=torch.bool)
            batch = torch.zeros_like(y, device=device)
            laplacian_eigenvector_pe = torch.randn((num_nodes, 10), device=device)
            random_walk_pe = torch.randn((num_nodes, 10), device=device)
            
            y_hat = model(x=x, 
                            edge_index=edge_index, 
                            edge_attr=edge_attr, 
                            y=y, 
                            label_mask=label_mask, 
                            batch=batch, 
                            laplacian_eigenvector_pe=laplacian_eigenvector_pe, 
                            random_walk_pe=random_walk_pe) 
            num_nodes += 1


def main(args):
    list_of_files = glob.glob(args.input_pdb)
    if len(args.input_pdb) < 1 or len(list_of_files) < 0:
        raise ValueError("Please fill input_pdb filename")
    
    if args.seed != -1:
        seed = args.seed    
    else:
        seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])

    seed = args.seed if args.seed != -1 else np.random.randint(0, high=42, size=1, dtype=int)[0]
    torch_geometric.seed_everything(seed)
    model_version = args.model_masked
    path_to_model = args.path_to_model
    if torch.cuda.is_available():
        device = torch.device('cuda')
        jit_model_path = 'model_scripted_fb_cuda.pt' if model_version else 'model_scripted_ml10_cuda.pt'
    else:
        device = torch.device('cpu')
        jit_model_path = 'model_scripted_fb_cpu.pt' if model_version else 'model_scripted_ml10_cpu.pt'
    jit_model_path = osp.join(path_to_model, jit_model_path)

    if args.iterative_sampling:generate = iterative_shot
    elif args.one_shot_diverse:generate = one_shot_diverse
    else:generate = one_shot_fast


    temperature = args.temperature
    iterative_sampling = args.iterative_sampling
    num_seq_per_target = args.num_seq_per_target
    one_shot_div = args.one_shot_diverse
    design_position = args.design_position
    verbose = args.verbose
    suppress = args.suppress_AAs
    
    if len(suppress) == 1 and suppress[0] == 'X':
        suppress = None
    else:
        alphabet = "".join(list(utils.aa1_to_idx.keys())) + 'X'
        suppress = torch.tensor([alphabet.index(letter) for letter in suppress], device=device)
                
    model = torch.jit.load(jit_model_path, map_location=device)
    model.to(device).eval()

    print('Model was loaded succesfully.')
    eol = False
    if len(list_of_files) >= 1:
        extra_outer_loop = tqdm(list_of_files, desc='Iteration through PDBs', position=0) if verbose else list_of_files
        eol = True
    dl = []
    if args.batched and len(list_of_files) > 2 and not iterative_sampling:
        if len(args.output_path) < 1:
            import os
            if not os.path.exists('output_GT_2'):
                os.makedirs('output_GT_2')
            output_path = 'output_GT_2'
        else:
            import os 
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)
            output_path = args.output_path
        generate = one_shot_fast_batch
        warmup_model(model, device)
        dl = []
        for input_pdb in extra_outer_loop:
            data = parse_and_featurize_light(pdb_path=input_pdb, design_position=design_position, device=device)
            data.pdb_path = input_pdb
            dl += [data]
        max_num = 25000
        skip_too_big = True
        samples =  []
        len_d = len(dl)
        indices = range(len_d)
        lengths = [dl[id].num_nodes for id in indices]
        sorted_ix = np.argsort(lengths)
        clusters  = []
        num_processed = 0
        current_num = 0
        while (num_processed < len_d):
            for i in sorted_ix[num_processed:]:
                data = dl[indices[i]]
                num = data.num_nodes
                if current_num + num > max_num:
                    if current_num == 0:
                        if skip_too_big:
                            continue
                    else:  # Mini-batch filled:
                        break

                samples.append(dl[indices[i]])
                num_processed += 1
                current_num += num
            clusters.append(samples)
            samples = []
            current_num = 0
        for clust in clusters:
            batch_data = Batch.from_data_list(clust)
            seq, sr, conf = generate(model, batch_data.to(device), temperature, verbose, suppress, eol, num_seq_per_target)
            ptr = batch_data.ptr.cpu()
            for i, pdb_path in enumerate(batch_data.pdb_path):
                name = pdb_path.split('/')[-1].split('.')[0]
                output_fasta = os.path.join(output_path,  name + '.fa')
                Y0 = S_to_seq(batch_data.y)
                with open(output_fasta, 'w') as fout:
                    fout.write('>' + name + '_init_sequece\n')
                    fout.write(Y0[ptr[i]:ptr[i+1]] + '\n')
                    for j, s in enumerate(seq):
                        s_res = s[ptr[i]:ptr[i+1]]
                        conf_res = conf[j][ptr[i]:ptr[i+1]].mean().item()
                        sr_res = seq_r(Y0[ptr[i]:ptr[i+1]],s_res)
                        fout.write('>gen_seq_' + str(j) + ' seq_recovery = ' + str(sr_res) + '; confidence = ' + str(conf_res) + '\n')
                        fout.write(s_res + '\n')

    else:
        for input_pdb in extra_outer_loop:
            
            if len(args.output_path) < 1:
                import os
                if not os.path.exists('output_GT'):
                    os.makedirs('output_GT')
                output_path = 'output_GT'
            else:
                import os 
                if not os.path.exists(args.output_path):
                    os.makedirs(args.output_path)
                output_path = args.output_path
            
            name = input_pdb.split('/')[-1].split('.')[0]
            output_fasta = os.path.join(output_path,  name + '.fa')

            data = parse_and_featurize_light(pdb_path=input_pdb, design_position=design_position, device=device)#, path=jit_featurizer_path)
            with open(output_fasta, 'w') as fout:
                fout.write('>' + name + '_init_sequece\n')
                fout.write(S_to_seq(data.y) + '\n')
                outer_loop = tqdm(range(num_seq_per_target), desc='Generation Sequences', position=1 if eol else 0, leave=False if eol else True) if verbose else range(num_seq_per_target)
                if iterative_sampling or one_shot_div:
                    for i in outer_loop:
                        if i > 0 and one_shot_div:
                            torch_geometric.seed_everything(seed + i)
                            data_tmp = parse_and_featurize_light(pdb_path=input_pdb, design_position=design_position, device=device, data=data_tmp)
                        else:
                            data_tmp = copy.copy(data)
                        data_tmp = data_tmp.to(device)
                        seq, sr, conf = generate(model, data_tmp, temperature, verbose, suppress, eol)
                        fout.write('>gen_seq_' + str(i) + ' seq_recovery = ' + str(sr) + '; confidence = ' + str(conf) + '\n')
                        fout.write(seq + '\n')        
                else:
                    data_tmp = copy.copy(data)
                    data_tmp = data_tmp.to(device)
                    seq, sr, conf = generate(model, data_tmp, temperature, verbose, suppress, eol, num_seq_per_target)
                    for i in outer_loop:
                        fout.write('>gen_seq_' + str(i) + ' seq_recovery = ' + str(sr[i]) + '; confidence = ' + str(conf[i]) + '\n')
                        fout.write(seq[i] + '\n')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_pdb", '-i', type=str, default='', help='PDB to design. Could be single file or path to folder')
    argparser.add_argument("--output_path", '-o', type=str, default='', help='Dir name for FASTA(s) with output sequence(s)')
    argparser.add_argument("--design_position", '-d', type=str, default='0', help='Position lists, e.g. 11 12 14:18. Default = 0 => design all residues')
    argparser.add_argument("--temperature", '-t' , type=float, default=0.1, help='Sampling temperature for amino acids. Higher values will lead to more diversity.')
    argparser.add_argument("--iterative_sampling", '-s',  action="store_true", default=False, help='Iterative decoding, otherwise one-shot generation')
    argparser.add_argument("--one_shot_diverse", '-D',  action="store_true", default=False, help='One-shot prediction with probably more diverse sequences')
    argparser.add_argument("--seed", type=int, default=-1, help="If set to -1 then a random seed will be picked")
    argparser.add_argument("--num_seq_per_target",'-n', type=int, default=1, help="Number of sequences to generate")
    argparser.add_argument("--suppress_AAs", '-x', type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    argparser.add_argument("--model_masked",'-m',  action="store_true", default=False, help="Using of model trained with masking")
    argparser.add_argument("--path_to_model", '-p', type=str, default='', help='Dir name with model_weights')
    argparser.add_argument("--verbose",'-v', action="store_true", default=False, help="Process detalisation level")
    argparser.add_argument("--batched",'-b', action="store_true", default=False, help="Experimental feature key for batched prediction")
    
    args = argparser.parse_args()  
    main(args)
   