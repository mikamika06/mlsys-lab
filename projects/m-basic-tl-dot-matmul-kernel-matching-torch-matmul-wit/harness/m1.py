import torch
import ref


def check(workdir):
    from triton_matmul.kernel import matmul_forward
    a, b = ref.get_test_matrices()
    try:
        out_tensor = matmul_forward(a, b)
    except Exception as e:
        return {"max_abs_err": 999.0, "_note": f"raised exception: {e}"}

    expected = torch.matmul(a, b)
    err = torch.max(torch.abs(out_tensor - expected)).item()
    return {"max_abs_err": float(err)}
