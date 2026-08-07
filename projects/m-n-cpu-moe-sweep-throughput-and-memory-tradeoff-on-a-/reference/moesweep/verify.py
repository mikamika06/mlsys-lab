def verify_tensor_placement(tensor_meta, override_tensors):
    verified = {}
    for name, meta in tensor_meta.items():
        is_expert = "mlp.experts" in name or "expert" in name
        should_be_cpu = name in override_tensors or (is_expert and override_tensors.get("all_experts", False))
        actual_device = meta.get("device", "gpu")
        expected_device = "cpu" if should_be_cpu else "gpu"
        verified[name] = (actual_device == expected_device)
    return all(verified.values()) if verified else False
