import torch
from mlsys import scorers


def _oracle(x, w1, b1, w2, b2):
    params = [x, w1, b1, w2, b2]
    h = torch.relu(x @ w1.t() + b1)
    y = h @ w2.t() + b2
    loss = y.sum()
    grads = torch.autograd.grad(loss, params)
    return [g.detach() for g in grads]


def grade(sol, fx) -> dict:
    torch.manual_seed(7)

    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    w1 = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    b1 = torch.randn(5, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    b2 = torch.randn(2, dtype=torch.float64, requires_grad=True)

    ref_grads = _oracle(x, w1, b1, w2, b2)

    try:
        loss, got_grads, saved_count = sol.checkpoint_segment(
            x, w1, b1, w2, b2
        )
    except Exception:
        return {"max_abs_err": float("inf"), "saved_tensor_count": -1}

    if len(got_grads) != len(ref_grads):
        return {"max_abs_err": float("inf"), "saved_tensor_count": int(saved_count)}

    errors = [
        scorers.max_abs_err(ref, got)
        for ref, got in zip(ref_grads, got_grads)
    ]

    recompute_inputs = [x, w1, b1, w2, b2]
    expected_saved = len(recompute_inputs)

    return {
        "max_abs_err": max(errors),
        "saved_tensor_count": int(saved_count) if isinstance(saved_count, int) else -1,
    }
