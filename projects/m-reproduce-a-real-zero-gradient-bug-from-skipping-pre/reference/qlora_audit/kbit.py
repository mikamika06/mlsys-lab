def prepare_model_for_kbit_training(model_config, prepare_kbit=True):
    config = dict(model_config)
    config["is_prepared_for_kbit"] = bool(prepare_kbit)
    if prepare_kbit:
        config["cast_inputs_to_fp32"] = True
        config["requires_grad_for_adapter"] = True
    else:
        config["cast_inputs_to_fp32"] = False
        config["requires_grad_for_adapter"] = False
    return config

def compute_adapter_gradients(model_state):
    prepared = model_state.get("is_prepared_for_kbit", False)
    has_adapter = model_state.get("has_adapter", False)
    trainable_params = model_state.get("trainable_params", [])

    grad_norms = {}
    if not has_adapter:
        for p in trainable_params:
            grad_norms[p] = 0.0
        return {"has_active_gradients": False, "grad_norms": grad_norms}

    if not prepared:
        for p in trainable_params:
            grad_norms[p] = 0.0
        return {"has_active_gradients": False, "grad_norms": grad_norms}

    for p in trainable_params:
        grad_norms[p] = 1.0
    return {"has_active_gradients": True, "grad_norms": grad_norms}
