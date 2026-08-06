def compute_mxfp4_share(model_spec):
    total_bytes = 0
    mxfp4_bytes = 0
    for layer in model_spec.get("layers", []):
        layer_type = layer.get("type", "dense")
        if layer_type == "moe":
            num_experts = layer.get("num_experts", 1)
            expert_params = layer.get("expert_params", 0)
            router_params = layer.get("router_params", 0)
            block_size = layer.get("block_size", 32)

            expert_bytes_per_expert = (expert_params * 4) // 8 + (expert_params // block_size) * 1
            total_expert_bytes = expert_bytes_per_expert * num_experts
            router_total_bytes = router_params * 2

            mxfp4_bytes += total_expert_bytes
            total_bytes += total_expert_bytes + router_total_bytes
        else:
            params = layer.get("params", 0)
            b_bytes = params * 2
            total_bytes += b_bytes

    if total_bytes == 0:
        return 0.0
    return float(mxfp4_bytes) / float(total_bytes)
