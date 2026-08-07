import importlib.util
import os
import numpy as np
import ref


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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_unmasked_causal": 0.0,
        "flops_match": 0.0
    }

    import sys
    sys.path.insert(0, workdir)

    try:
        from fused_attn.flops import compute_attention_flops, derive_tflops
        f_c = compute_attention_flops(2, 4, 128, 32, causal=True)
        f_nc = compute_attention_flops(2, 4, 128, 32, causal=False)
        tf = derive_tflops(2, 4, 128, 32, 0.005, causal=True)

        ref_f_c = ref.compute_attention_flops(2, 4, 128, 32, causal=True)
        ref_f_nc = ref.compute_attention_flops(2, 4, 128, 32, causal=False)
        ref_tf = ref.derive_tflops(2, 4, 128, 32, 0.005, causal=True)

        if f_c == ref_f_c and f_nc == ref_f_nc and abs(tf - ref_tf) < 1e-5:
            out["flops_match"] = 1.0
        else:
            out["_note"] = f"FLOPs mismatch: got ({f_c}, {f_nc}, {tf}), expected ({ref_f_c}, {ref_f_nc}, {ref_tf})"
    except Exception as e:
        out["_note"] = f"FLOPs calculation error: {type(e).__name__}: {str(e)}"
        return out

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on valid reference: {type(e).__name__}: {str(e)}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import fused_attn.causal as c_mod
    good_causal = c_mod.block_split_causal_attention_forward

    def broken_causal_unmasked(Q, K, V, sm_scale, block_size=32):
        import fused_attn.online_softmax as os_mod
        return os_mod.online_softmax_attention_forward(Q, K, V, sm_scale)

    c_mod.block_split_causal_attention_forward = broken_causal_unmasked
    try:
        out["catches_unmasked_causal"] = 0.0 if _survives(path) else 1.0
    finally:
        c_mod.block_split_causal_attention_forward = good_causal

    return out
