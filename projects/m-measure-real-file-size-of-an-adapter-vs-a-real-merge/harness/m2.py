import ref


def check(workdir):
    from adaptermerge.quantize import compute_quantization_error
    base, a_A, a_B = ref.get_sample_data()
    adapter_dict = {"layer.lora_A": a_A["layer.lora_A"], "layer.lora_B": a_B["layer.lora_B"]}

    want = ref.compute_quantization_error(base, adapter_dict)
    got = compute_quantization_error(base, adapter_dict)

    out = {"error_matched": 0.0}
    if isinstance(got, dict) and "error_merged" in got:
        if abs(got["error_merged"] - want["error_merged"]) < 1e-5:
            out["error_matched"] = 1.0
        else:
            out["_note"] = f"got error {got['error_merged']}, want {want['error_merged']}"
    else:
        out["_note"] = "compute_quantization_error did not return expected dict"
    return out
