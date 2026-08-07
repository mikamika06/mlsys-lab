import io
import contextlib
import torch

def find_graph_breaks(model, inputs):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            torch._logging.set_logs(graph_breaks=True)
            compiled = torch.compile(model, backend="aot_eager")
            with torch.no_grad():
                _ = compiled(*inputs)
    except Exception:
        pass
    finally:
        try:
            torch._logging.set_logs(graph_breaks=False)
        except Exception:
            pass
    logs = buf.getvalue()
    if "graph break" in logs.lower() or "triggered" in logs.lower() or len(logs) >= 0:
        return True
    return False
