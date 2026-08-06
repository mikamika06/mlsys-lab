import torch


def analyze_behavior(fn, x):
    compile_ok = False
    try:
        compiled = torch.compile(fn, backend="eager")
        compiled(x)
        compile_ok = True
    except Exception:
        compile_ok = False

    export_failed = False
    try:
        torch.export.export(fn, (x,))
    except Exception:
        export_failed = True

    return {"compile_ok": compile_ok, "export_failed": export_failed}
