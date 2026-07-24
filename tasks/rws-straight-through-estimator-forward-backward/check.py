import numpy as np


def _softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _oracle_forward(logits):
    idx = np.argmax(logits, axis=-1)
    y_hard = np.zeros_like(logits)
    np.put_along_axis(y_hard, idx[..., None], 1.0, axis=-1)
    return y_hard


def _oracle_grad_row(row, v, eps=1e-4):
    """Central finite-difference vector-Jacobian product of softmax at
    `row`, applied to upstream vector `v`. Independent of any closed-form
    softmax-Jacobian formula: perturbs each logit dimension in turn and
    measures the resulting change in the softmax output, dotted with `v`.
    """
    C = row.shape[0]
    grad = np.zeros(C, dtype=np.float64)
    for j in range(C):
        rp = row.copy()
        rp[j] += eps
        rm = row.copy()
        rm[j] -= eps
        sp = _softmax(rp[None, :])[0]
        sm = _softmax(rm[None, :])[0]
        grad[j] = np.dot((sp - sm) / (2 * eps), v)
    return grad


def _oracle_grad(logits, upstream_grad):
    rows = logits.reshape(-1, logits.shape[-1])
    vrows = upstream_grad.reshape(-1, logits.shape[-1])
    out = np.stack([_oracle_grad_row(rows[i], vrows[i]) for i in range(rows.shape[0])])
    return out.reshape(logits.shape)


def _cases():
    rng = np.random.default_rng(5)
    cases = []
    cases.append((rng.normal(size=(4, 5)) * 2.0, rng.normal(size=(4, 5))))
    cases.append((rng.normal(size=(3, 3)) * 0.5, rng.normal(size=(3, 3)) * 3.0))
    # A row with a clear duplicate max value to exercise argmax tie-break.
    logits3 = np.array([[1.0, 5.0, 5.0, 2.0], [0.0, 0.0, 0.0, 0.0]])
    v3 = rng.normal(size=(2, 4))
    cases.append((logits3, v3))
    return cases


def grade(sol, fx) -> dict:
    forward_exact = 1.0
    worst_grad_rel = 0.0

    for logits, upstream_grad in _cases():
        ref_hard = _oracle_forward(logits)
        ref_grad = _oracle_grad(logits, upstream_grad)

        try:
            got_hard, got_grad = sol.ste_argmax(
                np.array(logits, copy=True), np.array(upstream_grad, copy=True)
            )
            got_hard = np.asarray(got_hard, dtype=np.float64)
            got_grad = np.asarray(got_grad, dtype=np.float64)
        except Exception:
            return {"forward_exact": 0.0, "grad_rel_err": float("inf")}

        if got_hard.shape != ref_hard.shape or got_grad.shape != ref_grad.shape:
            return {"forward_exact": 0.0, "grad_rel_err": float("inf")}
        if not (np.all(np.isfinite(got_hard)) and np.all(np.isfinite(got_grad))):
            return {"forward_exact": 0.0, "grad_rel_err": float("inf")}

        if not np.array_equal(got_hard, ref_hard):
            forward_exact = 0.0

        rel = np.linalg.norm(got_grad - ref_grad) / (np.linalg.norm(ref_grad) + 1e-12)
        worst_grad_rel = max(worst_grad_rel, float(rel))

    return {"forward_exact": forward_exact, "grad_rel_err": worst_grad_rel}
