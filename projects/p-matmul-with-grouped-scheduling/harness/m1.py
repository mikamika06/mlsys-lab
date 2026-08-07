import torch
import ref


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from triton_matmul.kernel import matmul_basic

    m = {"compiles_and_runs": 0.0, "correctness": 0.0}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(128, 128, device=device)
    b = torch.randn(128, 128, device=device)

    try:
        out = matmul_basic(a, b)
        m["compiles_and_runs"] = 1.0
    except Exception:
        return m

    expected = ref.compute_oracle(a, b)
    if out is not None and out.shape == expected.shape:
        if torch.allclose(out, expected, atol=1e-2, rtol=1e-2):
            m["correctness"] = 1.0
    return m
