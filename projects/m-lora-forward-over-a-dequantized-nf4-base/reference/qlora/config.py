def fix_qlora_config(config):
    fixed = dict(config)

    quant_type = fixed.get("quant_type", "nf4")
    if quant_type not in ("nf4", "fp4"):
        fixed["quant_type"] = "nf4"

    bnb_4bit_compute_dtype = fixed.get("bnb_4bit_compute_dtype", "float16")
    torch_dtype = fixed.get("torch_dtype", "float16")

    if bnb_4bit_compute_dtype != torch_dtype:
        fixed["bnb_4bit_compute_dtype"] = torch_dtype
        fixed["has_dtype_mismatch_risk"] = True
    else:
        fixed["has_dtype_mismatch_risk"] = False

    r = fixed.get("r", 8)
    if r <= 0:
        fixed["r"] = 8
        r = 8

    alpha = fixed.get("lora_alpha", 16)
    fixed["scaling"] = float(alpha / r)

    return fixed
