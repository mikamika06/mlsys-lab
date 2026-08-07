import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_underallocation": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"tests fail on valid code: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "no test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import swapspace.sizing as sz

    good_seq = sz.compute_sequence_swap_bytes
    good_tot = sz.compute_total_swap_bytes

    def bad_seq(config, token_count):
        num_blocks = token_count // config["block_size"]
        blk = (
            2
            * config["num_layers"]
            * config["num_kv_heads"]
            * config["head_dim"]
            * config["block_size"]
            * config["dtype_bytes"]
        )
        return num_blocks * blk

    def bad_tot(config, token_counts):
        return sum(bad_seq(config, t) for t in token_counts)

    sz.compute_sequence_swap_bytes = bad_seq
    sz.compute_total_swap_bytes = bad_tot

    import swapspace

    swapspace.sizing.compute_sequence_swap_bytes = bad_seq
    swapspace.sizing.compute_total_swap_bytes = bad_tot

    try:
        out["catches_underallocation"] = 0.0 if _survives(path) else 1.0
    finally:
        sz.compute_sequence_swap_bytes = good_seq
        sz.compute_total_swap_bytes = good_tot
        swapspace.sizing.compute_sequence_swap_bytes = good_seq
        swapspace.sizing.compute_total_swap_bytes = good_tot

    return out
