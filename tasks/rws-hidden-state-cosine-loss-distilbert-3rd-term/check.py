import numpy as np

from mlsys import scorers


def _loss(h_t, h_s, eps):
    na = np.linalg.norm(h_t, axis=1)
    nb = np.linalg.norm(h_s, axis=1)
    dot = np.sum(h_t * h_s, axis=1)
    denom = na * nb + eps
    cos = dot / denom
    return float(np.mean(1.0 - cos))


def _grad_closed_form(h_t, h_s, eps):
    B = h_t.shape[0]
    na = np.linalg.norm(h_t, axis=1)
    nb = np.linalg.norm(h_s, axis=1)
    dot = np.sum(h_t * h_s, axis=1)
    denom = na * nb + eps
    term1 = h_t / denom[:, None]
    term2 = (dot * na / (nb * denom ** 2))[:, None] * h_s
    return -(term1 - term2) / B


def _grad_finite_diff(h_t, h_s, eps, step=1e-6):
    B, D = h_s.shape
    grad = np.zeros_like(h_s)
    for i in range(B):
        for j in range(D):
            hp = h_s.copy()
            hp[i, j] += step
            hm = h_s.copy()
            hm[i, j] -= step
            grad[i, j] = (_loss(h_t, hp, eps) - _loss(h_t, hm, eps)) / (2 * step)
    return grad


def _build_cases():
    cases = []
    for seed, B, D in [(0, 6, 5), (1, 4, 8), (2, 10, 3)]:
        rng = np.random.default_rng(seed)
        h_t = rng.standard_normal((B, D))
        h_s = rng.standard_normal((B, D)) * 0.7 + h_t * 0.3  # partially aligned
        cases.append((h_t, h_s, 1e-8))
    return cases


def grade(sol, fx) -> dict:
    worst_loss_rel = 0.0
    worst_grad_rel = 0.0

    for h_t, h_s, eps in _build_cases():
        loss_ref = _loss(h_t, h_s, eps)
        grad_fd = _grad_finite_diff(h_t, h_s, eps)
        # sanity: closed form must already agree with finite differences
        grad_cf = _grad_closed_form(h_t, h_s, eps)
        assert np.linalg.norm(grad_cf - grad_fd) / (np.linalg.norm(grad_fd) + 1e-12) < 1e-4

        try:
            out = sol.cosine_embedding_loss_and_grad(h_t.copy(), h_s.copy(), eps=eps)
            loss_got, grad_got = out
            loss_got = float(loss_got)
            grad_got = np.asarray(grad_got, dtype=np.float64)
        except Exception:
            return {"loss_rel_err": float("inf"), "grad_fd_rel_err": float("inf")}

        if grad_got.shape != h_s.shape or not (np.isfinite(loss_got) and np.all(np.isfinite(grad_got))):
            return {"loss_rel_err": float("inf"), "grad_fd_rel_err": float("inf")}

        loss_rel = abs(loss_got - loss_ref) / (abs(loss_ref) + 1e-12)
        worst_loss_rel = max(worst_loss_rel, loss_rel)
        worst_grad_rel = max(worst_grad_rel, scorers.rel_err(grad_fd, grad_got))

    return {"loss_rel_err": worst_loss_rel, "grad_fd_rel_err": worst_grad_rel}
