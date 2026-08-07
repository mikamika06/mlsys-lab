import torch
import ref


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from triton_matmul.kernel import matmul_basic

    m = {"edge_shapes_ok": 0.0}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shapes = [(33, 17, 49), (1, 1, 1), (257, 129, 65)]

    success = True
    for shape in shapes:
        a = torch.randn(shape[0], shape[2], device=device)
        b = torch.randn(shape[2], shape[1], device=device)
        try:
            out = matmul_basic(a, b)
            expected = ref.compute_oracle(a, b)
            if out.shape != expected.shape or not torch.allclose(out, expected, atol=1e-1, rtol=1e-1):
                success = False
                break
        except Exception:
            success = False
            break

    if success:
        m["edge_shapes_ok"] = 1.0
    return m
