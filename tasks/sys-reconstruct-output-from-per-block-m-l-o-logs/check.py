import numpy as np


def _local_block_stats(scores_block, v_block):
    m = float(np.max(scores_block))
    e = np.exp(scores_block - m)
    l = float(np.sum(e))
    o = e @ v_block
    return m, l, o


def _dense_reference(score_blocks, v_blocks):
    """Independent ground truth: plain global softmax attention computed
    directly over every raw score/value concatenated together -- NOT via
    any block-merge algebra, so it's a genuine check on whatever
    reconstruction method the student used."""
    scores = np.concatenate(score_blocks)
    values = np.concatenate(v_blocks, axis=0)
    m = np.max(scores)
    e = np.exp(scores - m)
    l = np.sum(e)
    o = e @ values
    return o / l


def _make_case(rng, block_sizes, d):
    score_blocks = [rng.normal(0.0, 3.0, size=bs) for bs in block_sizes]
    v_blocks = [rng.normal(0.0, 1.0, size=(bs, d)) for bs in block_sizes]

    block_m = np.empty(len(block_sizes), dtype=np.float64)
    block_l = np.empty(len(block_sizes), dtype=np.float64)
    block_o = np.empty((len(block_sizes), d), dtype=np.float64)
    for k, (sb, vb) in enumerate(zip(score_blocks, v_blocks)):
        m, l, o = _local_block_stats(sb, vb)
        block_m[k] = m
        block_l[k] = l
        block_o[k] = o

    ref = _dense_reference(score_blocks, v_blocks)
    return block_m, block_l, block_o, ref


def _cases():
    rng = np.random.default_rng(17)
    specs = [
        ([4, 4, 4, 4, 4], 4),          # uniform blocks
        ([3, 5, 2], 3),                 # few, uneven blocks
        ([1, 1, 1, 1, 1, 1], 5),        # singleton blocks (block_size=1)
        ([7, 2, 9, 1, 4], 6),           # ragged sizes
        ([6], 4),                        # single block (no merge needed)
    ]
    cases = []
    for block_sizes, d in specs:
        cases.append(_make_case(rng, block_sizes, d))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for block_m, block_l, block_o, ref in _cases():
        try:
            got = sol.reconstruct_attention_from_block_logs(
                block_m.copy(), block_l.copy(), block_o.copy()
            )
            got = np.asarray(got, dtype=np.float64).reshape(-1)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(got - ref)))
        worst = max(worst, err)

    return {"max_abs_err": worst}
