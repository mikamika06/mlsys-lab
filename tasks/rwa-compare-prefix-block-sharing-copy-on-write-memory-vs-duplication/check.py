import numpy as np

from mlsys import scorers


def _causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    d = q.shape[-1]
    n = q.shape[0]
    scores = (q @ k.T) / np.sqrt(d)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / np.sum(w, axis=-1, keepdims=True)
    return w @ v


def _ref_ratio(len_a: int, len_b: int, shared_prefix_len: int, block_size: int) -> float:
    blocks_a = -(-len_a // block_size)
    blocks_b = -(-len_b // block_size)
    shared_blocks = shared_prefix_len // block_size
    shared_blocks = min(shared_blocks, blocks_a, blocks_b)
    duplicated = blocks_a + blocks_b
    unique = duplicated - shared_blocks
    if unique == 0:
        return 0.0
    return float(duplicated / unique)


def _scenarios():
    rng = np.random.default_rng(0)
    scenarios = []

    def make(len_a, len_b, shared, block_size, d):
        shared = min(shared, len_a, len_b)
        q_a = rng.normal(size=(len_a, d))
        k_a = rng.normal(size=(len_a, d))
        v_a = rng.normal(size=(len_a, d))
        q_b = rng.normal(size=(len_b, d))
        k_b = rng.normal(size=(len_b, d))
        v_b = rng.normal(size=(len_b, d))
        if shared > 0:
            q_b[:shared] = q_a[:shared]
            k_b[:shared] = k_a[:shared]
            v_b[:shared] = v_a[:shared]
        return dict(
            q_a=q_a, k_a=k_a, v_a=v_a,
            q_b=q_b, k_b=k_b, v_b=v_b,
            shared_prefix_len=shared, block_size=block_size,
        )

    # exact-example scenario: identical, block size divides evenly minus one
    scenarios.append(make(len_a=3, len_b=3, shared=3, block_size=2, d=4))
    # block size that divides evenly
    scenarios.append(make(len_a=32, len_b=48, shared=16, block_size=16, d=8))
    # remainder blocks on both sides
    scenarios.append(make(len_a=37, len_b=51, shared=20, block_size=8, d=6))
    # zero shared prefix
    scenarios.append(make(len_a=20, len_b=25, shared=0, block_size=4, d=5))
    # shared prefix equals the full length of the shorter sequence
    scenarios.append(make(len_a=12, len_b=40, shared=12, block_size=5, d=3))
    # equal lengths, fully shared
    scenarios.append(make(len_a=64, len_b=64, shared=64, block_size=16, d=8))
    # block_size = 1 (every full-length shared run counts)
    scenarios.append(make(len_a=9, len_b=13, shared=7, block_size=1, d=4))
    # large-ish random scenario
    scenarios.append(make(len_a=100, len_b=130, shared=77, block_size=10, d=12))

    return scenarios


def grade(sol, fx) -> dict:
    worst_ratio_err = 0.0
    worst_max_abs_err = 0.0

    for sc in _scenarios():
        len_a = sc["q_a"].shape[0]
        len_b = sc["q_b"].shape[0]
        ref_ratio = _ref_ratio(len_a, len_b, sc["shared_prefix_len"], sc["block_size"])
        ref_out_a = _causal_attention(sc["q_a"], sc["k_a"], sc["v_a"])
        ref_out_b = _causal_attention(sc["q_b"], sc["k_b"], sc["v_b"])

        try:
            got_ratio, got_out_a, got_out_b = sol.cow_prefix_attention(
                sc["q_a"].copy(), sc["k_a"].copy(), sc["v_a"].copy(),
                sc["q_b"].copy(), sc["k_b"].copy(), sc["v_b"].copy(),
                sc["shared_prefix_len"], sc["block_size"],
            )
        except Exception:
            return {"ratio_err": float("inf"), "max_abs_err": float("inf")}

        try:
            got_ratio = float(got_ratio)
            got_out_a = np.asarray(got_out_a, dtype=np.float64)
            got_out_b = np.asarray(got_out_b, dtype=np.float64)
        except Exception:
            return {"ratio_err": float("inf"), "max_abs_err": float("inf")}

        if got_out_a.shape != ref_out_a.shape or got_out_b.shape != ref_out_b.shape:
            return {"ratio_err": float("inf"), "max_abs_err": float("inf")}

        ratio_err = abs(got_ratio - ref_ratio)
        err_a = scorers.max_abs_err(ref_out_a, got_out_a)
        err_b = scorers.max_abs_err(ref_out_b, got_out_b)

        worst_ratio_err = max(worst_ratio_err, ratio_err)
        worst_max_abs_err = max(worst_max_abs_err, err_a, err_b)

    return {"ratio_err": worst_ratio_err, "max_abs_err": worst_max_abs_err}
