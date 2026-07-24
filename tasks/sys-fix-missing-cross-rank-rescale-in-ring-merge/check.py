import numpy as np


def _make_partials(logit_blocks, values):
    partials = []
    for logits, v in zip(logit_blocks, values):
        m = np.max(logits, axis=1)
        e = np.exp(logits - m[:, None])
        l = np.sum(e, axis=1)
        a = e @ v
        partials.append((m, l, a))
    return partials


def _oracle(logit_blocks, values):
    logits = np.concatenate(logit_blocks, axis=1)
    vals = np.concatenate(values, axis=0)
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(z)
    p = p / np.sum(p, axis=1, keepdims=True)
    return p @ vals


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = [
        (
            [
                rng.normal(0, 1, size=(4, 3)),
                rng.normal(4, 1, size=(4, 5)),
                rng.normal(-2, 1, size=(4, 2)),
            ],
            [
                rng.normal(size=(3, 6)),
                rng.normal(size=(5, 6)),
                rng.normal(size=(2, 6)),
            ],
        ),
        (
            [
                rng.normal(10, 0.5, size=(3, 4)),
                rng.normal(-3, 0.5, size=(3, 4)),
            ],
            [
                rng.normal(size=(4, 3)),
                rng.normal(size=(4, 3)),
            ],
        ),
    ]

    worst = 0.0
    for logits, values in cases:
        partials = _make_partials(logits, values)
        ref = _oracle(logits, values)
        try:
            got = np.asarray(sol.ring_merge(partials), dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}
        err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
        worst = max(worst, float(err))
    return {"rel_err": worst}
