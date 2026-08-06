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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unscaled_lora": 0.0}

    try:
        from gguf_adapter.dequant import apply_lora_to_dequantized_base
        peft_dict, alpha, shapes = ref.generate_peft_data(seed=789)
        gguf_dict = ref.ref_convert_peft_to_gguf(peft_dict, alpha)
        deltas = {k: ref.ref_parse_and_build_delta(gguf_dict, k) for k in shapes}
        rng = np.random.default_rng(789)
        base_weights = {k: rng.standard_normal(v).astype(np.float32) for k, v in shapes.items()}

        got_fused = apply_lora_to_dequantized_base(base_weights, deltas)
        want_fused = ref.ref_apply_lora(base_weights, deltas)

        for k in base_weights:
            if not np.allclose(got_fused[k], want_fused[k], atol=1e-5):
                out["_note"] = f"apply_lora_to_dequantized_base failed numerical check on {k}"
                return out
    except Exception as e:
        out["_note"] = f"Error validating apply_lora_to_dequantized_base: {type(e).__name__}: {e}"
        return out

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"learner tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gguf_adapter.parser as parser_mod
    orig_parse = parser_mod.parse_lora_gguf_and_build_delta

    def faulty_parse(gguf_adapter_dict, target_layer_name):
        res = orig_parse(gguf_adapter_dict, target_layer_name)
        tensors = gguf_adapter_dict.get("tensors", {})
        mat_a = np.asarray(tensors[f"{target_layer_name}.lora_a"], dtype=np.float32)
        mat_b = np.asarray(tensors[f"{target_layer_name}.lora_b"], dtype=np.float32)
        res["delta"] = mat_b @ mat_a
        return res

    parser_mod.parse_lora_gguf_and_build_delta = faulty_parse
    import gguf_adapter
    gguf_adapter.parser.parse_lora_gguf_and_build_delta = faulty_parse

    try:
        out["catches_unscaled_lora"] = 0.0 if _survives(path) else 1.0
    finally:
        parser_mod.parse_lora_gguf_and_build_delta = orig_parse
        gguf_adapter.parser.parse_lora_gguf_and_build_delta = orig_parse

    return out
