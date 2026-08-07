import importlib.util
import os
import sys

sys.path.insert(0, ".")


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
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_uniform_fallback": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on reference code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import memrunner.predictor as predictor

    good_calc = predictor.calculate_weight_bytes

    def uniform_calc(config):
        num_layers = config["num_layers"]
        hidden_dim = config["hidden_dim"]
        intermediate_dim = config["intermediate_dim"]
        num_heads = config["num_heads"]
        num_kv_heads = config.get("num_kv_heads", num_heads)
        head_dim = hidden_dim // num_heads

        layer_params = (
            hidden_dim * (num_heads * head_dim)
            + hidden_dim * (num_kv_heads * head_dim)
            + hidden_dim * (num_kv_heads * head_dim)
            + (num_heads * head_dim) * hidden_dim
            + 2 * hidden_dim * intermediate_dim
            + intermediate_dim * hidden_dim
            + 2 * hidden_dim
        )
        vocab_size = config.get("vocab_size", 32000)
        total_params = layer_params * num_layers + 2 * vocab_size * hidden_dim
        return (total_params * 4.0) / 8.0

    predictor.calculate_weight_bytes = uniform_calc

    try:
        out["catches_uniform_fallback"] = 0.0 if _survives(path) else 1.0
    finally:
        predictor.calculate_weight_bytes = good_calc

    return out
