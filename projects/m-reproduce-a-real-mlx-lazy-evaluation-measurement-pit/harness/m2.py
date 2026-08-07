import ref
from mlxlora.sizes import calculate_model_sizes

def check(workdir):
    out = {"sizes_matched": 0.0, "ratio_matched": 0.0}

    model_cfg = ref.CONFIG_MODEL
    lora_cfg = ref.CONFIG_LORA

    want = ref.calculate_model_sizes(model_cfg, lora_cfg)
    got = calculate_model_sizes(model_cfg, lora_cfg)

    if (isinstance(got, dict) and
        got.get("base_bytes") == want["base_bytes"] and
        got.get("adapter_bytes") == want["adapter_bytes"]):
        out["sizes_matched"] = 1.0

    if (isinstance(got, dict) and
        abs(got.get("ratio", -1.0) - want["ratio"]) < 1e-6 and
        abs(got.get("adapter_percentage", -1.0) - want["adapter_percentage"]) < 1e-4):
        out["ratio_matched"] = 1.0

    if "_note" not in out and (out["sizes_matched"] == 0.0 or out["ratio_matched"] == 0.0):
        out["_note"] = f"got {got}, want {want}"

    return out
