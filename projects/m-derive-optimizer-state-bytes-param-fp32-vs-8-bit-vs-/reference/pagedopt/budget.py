def compute_qlora_budget(params: int, base_precision: str, lora_params: int, opt_type: str, opt_prec: str) -> int:
    from pagedopt.bytes import compute_optimizer_bytes
    if base_precision == "fp16":
        base_bytes = params * 2
    elif base_precision == "4-bit":
        base_bytes = int(params * 0.5 + params / 8.0)
    else:
        base_bytes = params * 4
    adapter_bytes = lora_params * 4
    grad_bytes = lora_params * 4
    optimizer_bytes = lora_params * compute_optimizer_bytes(opt_type, opt_prec)
    return int(base_bytes + adapter_bytes + grad_bytes + optimizer_bytes)
