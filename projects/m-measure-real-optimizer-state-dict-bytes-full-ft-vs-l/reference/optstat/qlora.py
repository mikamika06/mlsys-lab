def verify_qlora_optimizer_entries(state_dict, model_config):
    state = state_dict.get("state", {})
    frozen_param_ids = set()
    active_param_ids = set()

    for layer in model_config["layers"]:
        if layer.get("frozen", False):
            frozen_param_ids.add(layer["id"])
        else:
            active_param_ids.add(layer["id"])

    leaked = sorted([p_id for p_id in state if p_id in frozen_param_ids])
    active = sorted([p_id for p_id in state if p_id in active_param_ids])

    return {
        "is_valid": len(leaked) == 0,
        "leaked_param_ids": leaked,
        "active_param_ids": active,
    }
