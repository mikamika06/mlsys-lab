import numpy as np
from mlsys.scorers import max_abs_err

def _reference(emb):
    # Compute tied‑head logits using weight tying
    return emb.astype(np.float64) @ emb.T

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    vocab_size = 50
    dim = 32
    embeddings = rng.standard_normal((vocab_size, dim), dtype=np.float64)
    ref_logits = _reference(embeddings)
    try:
        cand_logits = sol.tied_head_logits(embeddings)
    except Exception:
        return {"max_abs_err": float("inf")}
    if cand_logits.shape != ref_logits.shape or cand_logits.dtype != np.float64:
        return {"max_abs_err": float("inf")}
    err = max_abs_err(ref_logits, cand_logits)
    return {"max_abs_err": err}
