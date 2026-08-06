def compute_tensor_breakdown(tensor_infos):
    breakdown = {}
    for name, size, block_type in tensor_infos:
        if "attn" in name or "attention" in name:
            cls = "attention"
        elif "router" in name or "gate" in name:
            cls = "router"
        elif "mlp" in name or "expert" in name or "ffn" in name:
            cls = "expert"
        else:
            cls = "other"

        breakdown.setdefault(cls, 0)
        breakdown[cls] += size
    return breakdown
