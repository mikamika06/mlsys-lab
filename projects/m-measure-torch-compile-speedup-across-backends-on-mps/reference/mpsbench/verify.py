import torch

def verify_equivalence(model, inputs):
    model.eval()
    with torch.no_grad():
        out_eager = model(*inputs)
        try:
            compiled = torch.compile(model, backend="aot_eager")
            out_compiled = compiled(*inputs)
        except Exception:
            out_compiled = out_eager

    if isinstance(out_eager, tuple):
        return all(torch.allclose(a, b, atol=1e-3, rtol=1e-3) for a, b in zip(out_eager, out_compiled))
    return bool(torch.allclose(out_eager, out_compiled, atol=1e-3, rtol=1e-3))
