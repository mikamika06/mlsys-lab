import numpy as np
import ref


def check(workdir):
    out = {
        "delta_reconstruction_matched": 0.0,
        "max_abs_err": 1.0
    }
    try:
        from gguf_adapter.parser import parse_lora_gguf_and_build_delta
    except Exception as e:
        out["_note"] = f"Failed to import parser: {type(e).__name__}: {e}"
        return out

    peft_dict, alpha, shapes = ref.generate_peft_data(seed=456)
    gguf_dict = ref.ref_convert_peft_to_gguf(peft_dict, alpha)

    max_err = 0.0
    matched = True

    for target_layer in shapes:
        want = ref.ref_parse_and_build_delta(gguf_dict, target_layer)
        try:
            got = parse_lora_gguf_and_build_delta(gguf_dict, target_layer)
        except Exception as e:
            out["_note"] = f"parse_lora_gguf_and_build_delta failed on {target_layer}: {type(e).__name__}: {e}"
            return out

        if not isinstance(got, dict) or "delta" not in got:
            out["_note"] = f"Parsed dict for {target_layer} missing 'delta' key."
            return out

        err = float(np.max(np.abs(got["delta"] - want["delta"])))
        if err > max_err:
            max_err = err

        if got.get("rank") != want["rank"] or not np.isclose(got.get("scaling", -1), want["scaling"]):
            matched = False

    out["max_abs_err"] = max_err
    if matched and max_err <= 1e-5:
        out["delta_reconstruction_matched"] = 1.0

    return out
