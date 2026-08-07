def select_torchao_config(constraint_dict: dict) -> str:
    max_mse = constraint_dict.get("max_mse", 0.01)
    allow_dynamic = constraint_dict.get("allow_dynamic", True)
    if max_mse < 0.001 and allow_dynamic:
        return "int8_dynamic_activation_int8_weight"
    elif not allow_dynamic:
        return "int8_weight_only"
    else:
        return "int8_weight_only"
