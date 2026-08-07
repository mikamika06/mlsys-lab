import ref
import torch

def check(workdir):
    from backends.minimal import register_minimal_backend
    out = {"backend_runs": 0.0}
    try:
        model = ref.SimpleModel()
        x = torch.randn(2, 16)
        compiled_fn = register_minimal_backend(model)
        res = compiled_fn(x)
        expected = model(x)
        if torch.allclose(res, expected, atol=1e-5):
            out["backend_runs"] = 1.0
        else:
            out["_note"] = "backend output did not match expected eager output"
    except Exception as e:
        out["_note"] = f"error running minimal backend: {str(e)[:120]}"
    return out
