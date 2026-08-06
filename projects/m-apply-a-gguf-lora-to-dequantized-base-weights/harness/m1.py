import numpy as np
import ref


def check(workdir):
    out = {"conversion_correct": 0.0}
    try:
        from adapter.convert import convert_peft_to_gguf
    except Exception as e:
        out["_note"] = f"Failed to import convert_peft_to_gguf: {type(e).__name__}: {e}"
        return out

    peft_dict, alpha, _ = ref.generate_peft_data(seed=123)
    want_gguf = ref.ref_convert_peft_to_gguf(peft_dict, alpha)

    try:
        got_gguf = convert_peft_to_gguf(peft_dict, alpha)
    except Exception as e:
        out["_note"] = f"convert_peft_to_gguf raised: {type(e).__name__}: {e}"
        return out

    if not isinstance(got_gguf, dict) or "metadata" not in got_gguf or "tensors" not in got_gguf:
        out["_note"] = "Returned GGUF structure missing 'metadata' or 'tensors' keys."
        return out

    if float(got_gguf["metadata"].get("adapter.lora.alpha", -1.0)) != alpha:
        out["_note"] = f"Alpha mismatch. Expected {alpha}, got {got_gguf['metadata'].get('adapter.lora.alpha')}"
        return out

    got_tensors = got_gguf["tensors"]
    want_tensors = want_gguf["tensors"]

    if set(got_tensors.keys()) != set(want_tensors.keys()):
        out["_note"] = f"Tensor keys mismatch. Got {sorted(got_tensors.keys())}, expected {sorted(want_tensors.keys())}"
        return out

    for k in want_tensors:
        if not np.allclose(got_tensors[k], want_tensors[k]):
            out["_note"] = f"Tensor content mismatch for key {k}"
            return out

    out["conversion_correct"] = 1.0
    return out
