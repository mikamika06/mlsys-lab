import importlib.util
import os
import sys


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
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_kv_flops": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

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

    import roofline.flops as f_mod
    good_flops = f_mod.compute_prefill_flops_per_token

    def faulty_flops(config, seq_len):
        h = config["hidden_size"]
        n_layers = config["num_hidden_layers"]
        n_heads = config["num_attention_heads"]
        d_head = h // n_heads
        i = config.get("intermediate_size", 4 * h)

        q_flops = 2 * h * (n_heads * d_head)
        out_flops = 2 * (n_heads * d_head) * h
        attn_score_flops = 2 * n_heads * d_head * seq_len
        attn_val_flops = 2 * n_heads * seq_len * d_head
        attn_flops = q_flops + out_flops + attn_score_flops + attn_val_flops

        mlp_flops = 3 * (2 * h * i)
        return float((attn_flops + mlp_flops) * n_layers)

    f_mod.compute_prefill_flops_per_token = faulty_flops
    import roofline
    roofline.compute_prefill_flops_per_token = faulty_flops

    try:
        out["catches_missing_kv_flops"] = 0.0 if _survives(path) else 1.0
    finally:
        f_mod.compute_prefill_flops_per_token = good_flops
        roofline.compute_prefill_flops_per_token = good_flops

    return out
