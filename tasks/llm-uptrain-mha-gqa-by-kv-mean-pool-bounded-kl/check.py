import numpy as np
from mlsys import scorers


def _oracle_pool(k, v, groups):
    b, h, tk, d = k.shape
    per_group = h // groups
    k_ref = k.reshape(b, groups, per_group, tk, d).mean(axis=2)
    v_ref = v.reshape(b, groups, per_group, tk, d).mean(axis=2)
    return k_ref, v_ref


def _as_logits(k, v):
    return np.concatenate(
        [
            k.reshape(k.shape[0], -1),
            v.reshape(v.shape[0], -1),
        ],
        axis=-1,
    )


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(123)
    cases = [
        (1, 8, 4, 5, 16, 4),
        (2, 6, 3, 4, 8, 3),
        (1, 4, 2, 6, 12, 2),
    ]

    worst = 0.0
    for b, h, tq, tk, d, groups in cases:
        q = rng.normal(size=(b, h, tq, d)).astype(np.float64)
        k = rng.normal(size=(b, h, tk, d)).astype(np.float64)
        v = rng.normal(size=(b, h, tk, d)).astype(np.float64)

        try:
            kg, vg = sol.uptrain_mha_to_gqa(q, k, v, groups)
        except Exception:
            return {"mean_kl": float("inf")}

        ref_k, ref_v = _oracle_pool(k, v, groups)

        if kg.shape != ref_k.shape or vg.shape != ref_v.shape:
            return {"mean_kl": float("inf")}

        ref_logits = _as_logits(ref_k, ref_v)
        cand_logits = _as_logits(np.asarray(kg), np.asarray(vg))

        value = scorers.mean_kl(ref_logits, cand_logits)
        if not np.isfinite(value):
            return {"mean_kl": float("inf")}
        worst = max(worst, value)

    return {"mean_kl": float(worst)}
