import ref
import torch


def check(workdir):
    from gradclip.clip import compute_grad_norm, clip_grads
    out = {"norm_match": 0.0, "scaled_match": 0.0, "rel_err": 1.0}
    params = ref.get_test_cases()[0]

    p_ref = [torch.nn.Parameter(p.clone()) for p in params]
    for i, p in enumerate(p_ref):
        p.grad = params[i].grad.clone()

    p_got = [torch.nn.Parameter(p.clone()) for p in params]
    for i, p in enumerate(p_got):
        p.grad = params[i].grad.clone()

    want_norm = ref.compute_grad_norm(p_ref, norm_type=2.0)
    got_norm = compute_grad_norm(p_got, norm_type=2.0)

    ref.clip_grads(p_ref, max_norm=1.0, norm_type=2.0)
    clip_grads(p_got, max_norm=1.0, norm_type=2.0)

    diff = torch.abs(got_norm - want_norm).item() / (torch.abs(want_norm).item() + 1e-8)
    out["rel_err"] = float(diff)
    if diff < 1e-5:
        out["norm_match"] = 1.0

    matched = True
    for pr, pg in zip(p_ref, p_got):
        if not torch.allclose(pr.grad, pg.grad, atol=1e-5, rtol=1e-5):
            matched = False
            break

    if matched:
        out["scaled_match"] = 1.0

    return out
