from optmem.states import compute_optimizer_bytes


def compare_memory(config):
    """Compare memory between full fine-tuning and LoRA."""
    full_res = compute_optimizer_bytes(config, "full")
    lora_res = compute_optimizer_bytes(config, "lora")
    ratio = lora_res["total"] / full_res["total"]
    diff = full_res["total"] - lora_res["total"]
    return {
        "full_total": full_res["total"],
        "lora_total": lora_res["total"],
        "ratio": ratio,
        "absolute_savings": diff,
    }
