import ref
import torch


def check(workdir):
    try:
        from triton_autograd.kernels import triton_silu_forward
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"failed to import triton_silu_forward: {e}"}

    max_err = 0.0
    for x, _ in ref.get_test_tensors():
        try:
            got = triton_silu_forward(x)
            want = ref.torch_silu_forward(x)
            diff = torch.norm(got - want)
            norm = torch.norm(want)
            err = (diff / (norm + 1e-8)).item()
            if err > max_err:
                max_err = err
        except Exception as e:
            return {"rel_err": 1.0, "_note": f"triton_silu_forward execution failed: {e}"}

    return {"rel_err": float(max_err)}
