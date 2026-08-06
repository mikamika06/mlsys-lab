import importlib.util
import os
import torch


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_swapped_fg": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests failed on baseline: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import tp.operators as op
    orig_col_fwd = op.ColumnParallelMatmulFunction.forward
    orig_row_fwd = op.RowParallelMatmulFunction.forward

    class SwappedCol(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_, weight, process_group=None):
            ctx.save_for_backward(input_, weight)
            return torch.matmul(input_, weight.t())

        @staticmethod
        def backward(ctx, grad_output):
            input_, weight = ctx.saved_tensors
            grad_input = torch.matmul(grad_output, weight) * 2.0
            grad_weight = torch.matmul(grad_output.transpose(-2, -1), input_)
            return grad_input, grad_weight, None

    def broken_col(input_, weight, process_group=None):
        return SwappedCol.apply(input_, weight, process_group)

    op.column_parallel_matmul = broken_col

    try:
        out["catches_swapped_fg"] = 0.0 if _survives(path) else 1.0
    finally:
        op.column_parallel_matmul = op.ColumnParallelMatmulFunction.apply
