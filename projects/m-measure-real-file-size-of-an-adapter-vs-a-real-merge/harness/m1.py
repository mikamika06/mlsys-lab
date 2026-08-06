import os
import tempfile
import ref


def check(workdir):
    from adaptermerge.measure import measure_file_sizes
    base, a_A, a_B = ref.get_sample_data()
    adapter_dict = {"layer.lora_A": a_A["layer.lora_A"], "layer.lora_B": a_B["layer.lora_B"]}

    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        want = ref.measure_file_sizes(base, adapter_dict, tmp1)
        got = measure_file_sizes(base, adapter_dict, tmp2)

    out = {"size_ratio_matched": 0.0}
    if isinstance(got, dict) and "size_ratio" in got:
        if abs(got["size_ratio"] - want["size_ratio"]) < 1e-5:
            out["size_ratio_matched"] = 1.0
        else:
            out["_note"] = f"got ratio {got['size_ratio']}, want {want['size_ratio']}"
    else:
        out["_note"] = "measure_file_sizes did not return a dict with 'size_ratio'"
    return out
