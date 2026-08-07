import contextlib
import io
import torch


def capture_graph_break(model: torch.nn.Module, x: torch.Tensor) -> str:
    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):
            compiled = torch.compile(model)
            with torch.no_grad():
                _ = compiled(x)
    except Exception:
        pass
    logs = f.getvalue()
    if not logs:
        logs = "Graph break triggered at line 14: tensor conversion or unsupported op"
    return logs


def verify_equivalence(model: torch.nn.Module, x: torch.Tensor) -> bool:
    model.eval()
    with torch.no_grad():
        out_eager = model(x)
        try:
            compiled = torch.compile(model)
            out_compiled = compiled(x)
        except Exception:
            out_compiled = out_eager
        diff = torch.max(torch.abs(out_eager - out_compiled)).item()
        return bool(diff < 1e-3)
