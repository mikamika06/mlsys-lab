import importlib.util
import os
import sys


def _run(path):
    sys.modules.pop("learner_regression", None)
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_alignment_checks": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import blockalign.validator as val

    good_val = val.validate_block_size
    good_filter = val.filter_valid_block_sizes

    def buggy_val_no_multiple(backend, model, block_size):
        b = dict(backend)
        b["block_multiple"] = 1
        return good_val(b, model, block_size)

    def buggy_filter_no_multiple(backend, model, candidate_sizes):
        return [bs for bs in candidate_sizes if buggy_val_no_multiple(backend, model, bs)["valid"]]

    def buggy_val_no_quant(backend, model, block_size):
        m = dict(model)
        m["is_quantized"] = False
        return good_val(backend, m, block_size)

    def buggy_filter_no_quant(backend, model, candidate_sizes):
        return [bs for bs in candidate_sizes if buggy_val_no_quant(backend, model, bs)["valid"]]

    catches = 0

    val.validate_block_size = buggy_val_no_multiple
    val.filter_valid_block_sizes = buggy_filter_no_multiple
    if "blockalign.planner" in sys.modules:
        sys.modules["blockalign.planner"].validate_block_size = buggy_val_no_multiple
        sys.modules["blockalign.planner"].filter_valid_block_sizes = buggy_filter_no_multiple

    if not _survives(path):
        catches += 1

    val.validate_block_size = buggy_val_no_quant
    val.filter_valid_block_sizes = buggy_filter_no_quant
    if "blockalign.planner" in sys.modules:
        sys.modules["blockalign.planner"].validate_block_size = buggy_val_no_quant
        sys.modules["blockalign.planner"].filter_valid_block_sizes = buggy_filter_no_quant

    if not _survives(path):
        catches += 1

    val.validate_block_size = good_val
    val.filter_valid_block_sizes = good_filter
    if "blockalign.planner" in sys.modules:
        sys.modules["blockalign.planner"].validate_block_size = good_val
        sys.modules["blockalign.planner"].filter_valid_block_sizes = good_filter

    if catches == 2:
        out["catches_missing_alignment_checks"] = 1.0
    else:
        out["_note"] = f"tests caught {catches}/2 alignment bugs"

    return out
