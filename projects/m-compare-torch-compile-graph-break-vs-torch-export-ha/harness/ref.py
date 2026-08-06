import torch


def sample_branch_fn(x):
    if x.sum() > 0:
        return x * 2.0
    else:
        return x * 3.0


def check_analysis(fn, x):
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


def true_branch(x):
    return x * 2.0


def false_branch(x):
    return x * 3.0


def check_cond_fn(fn, x):
    try:
        res = fn(x)
        exported = torch.export.export(fn, (x,))
        return {"result": res, "exported": exported is not None}
    except Exception:
        return {"result": None, "exported": False}
