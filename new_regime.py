import numpy as np
import torch
import torch.nn.functional as F

AA_LIST = list("ARNDCQEGHILKMFPSTWYV")
AA_TO_IDX = {a: i for i, a in enumerate(AA_LIST)}

# Optional: Biopython BLOSUM62 loader
try:
    from Bio.Align import substitution_matrices
    BLOSUM62 = substitution_matrices.load("BLOSUM62")
    HAVE_BIOPYTHON = True
except Exception:
    HAVE_BIOPYTHON = False
    BLOSUM62 = None


# ============================================================
# Basic helpers
# ============================================================

def blosum_score(a, b):
    if not HAVE_BIOPYTHON:
        raise ImportError("Biopython is not installed. pip install biopython")
    return float(BLOSUM62[a, b])

def seq_blosum_score(seq, ref_seq):
    assert len(seq) == len(ref_seq)
    s = 0.0
    for a, b in zip(seq, ref_seq):
        s += blosum_score(a, b)
    return s / len(seq)

def model_logprob_score(seq, logits):
    log_probs = F.log_softmax(logits, dim=-1)
    s = 0.0
    for i, aa in enumerate(seq):
        s += log_probs[i, AA_TO_IDX[aa]].item()
    return s / len(seq)

def sequence_recovery(seq, native_seq):
    return sum(a == b for a, b in zip(seq, native_seq)) / len(seq)

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

def idx_to_seq(idx_list):
    return "".join(AA_LIST[i] for i in idx_list)

def seq_to_idx(seq):
    return [AA_TO_IDX[a] for a in seq]


# ============================================================
# Position selection: choose mutable residues by entropy
# ============================================================

def position_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    return ent  # [L]

def choose_mutable_positions(logits, mutable_frac=0.30):
    ent = position_entropy(logits)
    L = logits.shape[0]
    k = max(1, int(L * mutable_frac))
    mutable_idx = torch.topk(ent, k=k).indices.tolist()
    return set(mutable_idx)


# ============================================================
# Sampling: adaptive temperature + freeze + BLOSUM bias
# ============================================================

def sample_sequence_hybrid(
    logits,
    native_seq=None,
    template_seq=None,
    mutable_frac=0.30,
    freeze_threshold=0.90,
    T_min=0.10,
    alpha=0.50,
    top_k=5,
    blosum_beta=0.10,
    return_metrics=False,
):
    """
    logits: [L, 20]

    Returns:
        seq
        or (seq, metrics)
    """
    L = logits.shape[0]
    probs = F.softmax(logits, dim=-1)
    max_prob, greedy_idx = probs.max(dim=-1)

    mutable_set = choose_mutable_positions(logits, mutable_frac=mutable_frac)
    T_pos = T_min + alpha * (1.0 - max_prob)

    seq_idx = []

    # metrics collected per position
    chosen_probs = []
    pos_entropies = []

    for i in range(L):
        # Freeze highly confident positions
        if max_prob[i].item() >= freeze_threshold:
            aa_idx = int(greedy_idx[i].item())
            seq_idx.append(aa_idx)

            chosen_probs.append(float(probs[i, aa_idx].item()))
            pos_entropies.append(0.0)
            continue

        # Optionally restrict sampling to uncertain positions only
        if i not in mutable_set:
            if native_seq is not None:
                aa_idx = AA_TO_IDX[native_seq[i]]
            else:
                aa_idx = int(greedy_idx[i].item())

            seq_idx.append(aa_idx)

            chosen_probs.append(float(probs[i, aa_idx].item()))
            pos_entropies.append(0.0)
            continue

        T_i = T_pos[i].item()
        base_logits = logits[i] / T_i  # [20]

        # Soft BLOSUM bias
        if template_seq is not None:
            ref_aa = template_seq[i]
            blosum_vec = torch.tensor(
                [blosum_score(ref_aa, aa) for aa in AA_LIST],
                dtype=base_logits.dtype,
                device=base_logits.device,
            )
            combined = base_logits + blosum_beta * blosum_vec
        else:
            combined = base_logits

        p = F.softmax(combined, dim=-1)
        ent_i = float((-(p * p.clamp_min(1e-12).log()).sum()).item())

        vals, idx = torch.topk(p, top_k)
        vals = vals / vals.sum()
        sampled = idx[torch.multinomial(vals, 1)].item()
        seq_idx.append(int(sampled))

        chosen_probs.append(float(p[sampled].item()))
        pos_entropies.append(ent_i)

    seq = idx_to_seq(seq_idx)

    if not return_metrics:
        return seq

    metrics = {
        "confidence": float(np.mean(chosen_probs)),
        "entropy": float(np.mean(pos_entropies)),
        "mean_logprob": float(model_logprob_score(seq, logits)),
    }
    if native_seq is not None:
        metrics["recovery"] = float(sequence_recovery(seq, native_seq))

    return seq, metrics


def generate_pool(
    logits,
    n_samples=3000,
    native_seq=None,
    template_seq=None,
    mutable_frac=0.30,
    freeze_threshold=0.90,
    T_min=0.10,
    alpha=0.50,
    top_k=5,
    blosum_beta=0.10,
):
    """
    Returns list of records:
      [{"seq": ..., "confidence": ..., "entropy": ..., ...}, ...]
    """
    pool = []
    seen = set()

    for _ in range(n_samples):
        seq, metrics = sample_sequence_hybrid(
            logits=logits,
            native_seq=native_seq,
            template_seq=template_seq,
            mutable_frac=mutable_frac,
            freeze_threshold=freeze_threshold,
            T_min=T_min,
            alpha=alpha,
            top_k=top_k,
            blosum_beta=blosum_beta,
            return_metrics=True,
        )

        if seq in seen:
            continue
        seen.add(seq)

        rec = {
            "seq": seq,
            "confidence": metrics["confidence"],
            "entropy": metrics["entropy"],
            "mean_logprob": metrics["mean_logprob"],
        }
        if "recovery" in metrics:
            rec["recovery"] = metrics["recovery"]

        pool.append(rec)

    return pool


# ============================================================
# Reranking: model score + BLOSUM + recovery floor
# ============================================================

def combined_score(seq, logits, ref_seq=None, lambda_blosum=0.15):
    score = model_logprob_score(seq, logits)
    if ref_seq is not None:
        score += lambda_blosum * seq_blosum_score(seq, ref_seq)
    return score


def filter_by_recovery(pool, native_seq, min_recovery=0.80):
    kept = []
    for item in pool:
        rec = item.get("recovery", sequence_recovery(item["seq"], native_seq))
        if rec >= min_recovery:
            item = dict(item)
            item["recovery"] = rec
            kept.append(item)
    return kept


# ============================================================
# Diversity selection: Hamming or BLOSUM-aware
# ============================================================

def select_diverse_sequences(
    pool,
    scores,
    n_final=50,
    use_blosum_distance=False,
    alpha=0.70,
):
    """
    pool: list[str]
    scores: list[float]
    """
    if len(pool) == 0:
        return []

    ranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)

    selected = [ranked[0][0]]
    candidates = [x[0] for x in ranked[1:]]

    while len(selected) < n_final and candidates:
        best_seq = None
        best_obj = -1e9

        for seq in candidates:
            q = dict(ranked).get(seq, 0.0)

            if use_blosum_distance:
                min_dist = min(-seq_blosum_score(seq, s) for s in selected)
            else:
                min_dist = min(hamming(seq, s) for s in selected)

            obj = alpha * q + (1.0 - alpha) * min_dist

            if obj > best_obj:
                best_obj = obj
                best_seq = seq

        selected.append(best_seq)
        candidates.remove(best_seq)

    return selected


# ============================================================
# Full hybrid pipeline
# ============================================================

def design_pipeline_hybrid(
    logits,
    native_seq=None,
    template_seq=None,
    n_samples=5000,
    keep_top=800,
    n_final=100,
    min_recovery=0.80,
    mutable_frac=0.35,
    freeze_threshold=0.92,
    T_min=0.10,
    alpha_temp=0.45,
    top_k=4,
    blosum_beta=0.08,
    lambda_blosum=0.10,
    alpha_diversity=0.70,
    use_blosum_distance=False,
    native_seq_for_out = None,
):
    """
    Returns:
        final_records: list[dict]
    """
    pool = generate_pool(
        logits=logits,
        n_samples=n_samples,
        native_seq=native_seq,
        template_seq=template_seq,
        mutable_frac=mutable_frac,
        freeze_threshold=freeze_threshold,
        T_min=T_min,
        alpha=alpha_temp,
        top_k=top_k,
        blosum_beta=blosum_beta,
    )

    # Recovery filter
    if native_seq is not None:
        pool = filter_by_recovery(pool, native_seq, min_recovery=min_recovery)

    if len(pool) == 0:
        raise ValueError(
            "No candidates passed the recovery floor. "
            "Try lowering min_recovery, lowering T_min / alpha_temp, "
            "or decreasing mutable_frac."
        )

    # Score candidates
    scores = [
        combined_score(item["seq"], logits, ref_seq=template_seq, lambda_blosum=lambda_blosum)
        for item in pool
    ]

    # Keep best quality subset
    ranked_items = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)[:keep_top]
    top_pool = [x[0]["seq"] for x in ranked_items]
    top_scores = [x[1] for x in ranked_items]

    # Diversity reranking
    final_sequences = select_diverse_sequences(
        pool=top_pool,
        scores=top_scores,
        n_final=n_final,
        use_blosum_distance=use_blosum_distance,
        alpha=alpha_diversity,
    )

    # Build output records
    top_lookup = {item["seq"]: item for item, _ in ranked_items}
    final_records = []

    for seq in final_sequences:
        item = dict(top_lookup[seq])
        item["quality"] = combined_score(seq, logits, ref_seq=template_seq, lambda_blosum=lambda_blosum)
        if native_seq_for_out is not None:
            item["recovery"] = sequence_recovery(seq, native_seq_for_out)
        final_records.append(item)

    # Report
    for i, item in enumerate(final_records[:10]):
        line = f"{i+1:02d} | quality={item['quality']:.4f} | conf={item['confidence']:.4f} | ent={item['entropy']:.4f}"
        if native_seq_for_out is not None:
            line += f" | recovery={item['recovery']:.3f}"
        line += f" | {item['seq'][:60]}"
        # print(line)

    if native_seq_for_out is not None and len(final_records) > 0:
        recs = [x["recovery"] for x in final_records]
        # print(
        #     f"Mean recovery: {np.mean(recs):.3f} | "
        #     f"Median recovery: {np.median(recs):.3f} | "
        #     f"Min recovery: {np.min(recs):.3f}"
        # )

    return final_records
def kak_b_sempler_logits(model, data, temperature=1.0):
    with torch.no_grad():
        data.label_mask = torch.zeros_like(data.y, dtype=torch.bool)
        logits = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y, label_mask=data.label_mask, batch=torch.zeros_like(data.y), laplacian_eigenvector_pe=data.laplacian_eigenvector_pe, random_walk_pe=data.random_walk_pe) 
        for _ in range(3):
            data.label_mask = torch.zeros_like(data.y, dtype=torch.bool)
            mask = torch.randint(0, data.y.shape[0], (int(torch.rand(1) * data.y.shape[0]),))
            data.label_mask[mask] = True
            data.y[~mask] = torch.distributions.Categorical(probs=(logits / temperature).softmax(dim=-1)).sample()[~mask]
            # data.y = torch.distributions.Categorical(probs=(logits / temperature).softmax(dim=-1)).sample()
            logits = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y, label_mask=data.label_mask, batch=torch.zeros_like(data.y), laplacian_eigenvector_pe=data.laplacian_eigenvector_pe, random_walk_pe=data.random_walk_pe) 
    return logits

from utils import S_to_seq
def sampler_refinement(data_input, model, temperature, num_samples):
    Y = data_input.y.clone()
    data = data_input.clone()
    
    ll = int(num_samples ** 0.5) + 1
    ss = ll
    FR = []
    for i in range(ll):
        logits = kak_b_sempler_logits(model, data, temperature=temperature).detach().cpu()
        final_records = design_pipeline_hybrid(
            logits=logits,
            native_seq=None,
            template_seq=None,   
            n_samples=1000,
            keep_top=100,
            n_final=ss,
            min_recovery=0.80,
            mutable_frac=0.35,
            freeze_threshold=0.8,
            T_min=temperature,
            alpha_temp=0.45,
            top_k=10,
            blosum_beta=0.12,
            lambda_blosum=0.20,
            alpha_diversity=0.45,
            use_blosum_distance=False,
            native_seq_for_out=S_to_seq(Y),
            
        )
        FR = FR + final_records
    return FR[:num_samples]
