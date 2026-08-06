import torch
import ref


def check(workdir):
    from lora.grad import compute_lora_gradients

    torch.manual_seed(42)
    batch_size = 4
    in_f = 32
    out_f = 16
    r = 4
    alpha = 8.0

    X = torch.randn(batch_size, in_f, requires_grad=True)
    W = torch.randn(out_f, in_f, requires_grad=True)
    A = torch.randn(r, in_f, requires_grad=True)
    B = torch.randn(out_f, r, requires_grad=True)
    dL_dY = torch.randn(batch_size, out_f)

    # Reference gradients via PyTorch autograd
    scaling = alpha / r
    W_eff = W + (B @ A) * scaling
    Y = X @ W_eff.T
    Y.backward(dL_dY)

    ref_dA = A.grad.clone()
    ref_dB = B.grad.clone()

    got_dA, got_dB = compute_lora_gradients(X.detach(), W.detach(), A.detach(), B.detach(), alpha, dL_dY)

    err_A = torch.max(torch.abs(got_dA - ref_dA)).item()
    err_B = torch.max(torch.abs(got_dB - ref_dB)).item()
    max_err = max(err_A, err_B)

    return {"max_abs_err": float(max_err)}
