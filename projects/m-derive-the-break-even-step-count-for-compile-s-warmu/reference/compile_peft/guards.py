def classifies_guard_failure(change_name):
    triggers = {
        "tensor_shape_change": True,
        "tensor_dtype_change": True,
        "python_global_control_flow": True,
        "optimizer_lr_update_in_place": False,
        "tensor_device_change": True,
        "weight_tensor_data_mutation": False,
    }
    if change_name not in triggers:
        raise ValueError(f"Unknown change: {change_name}")
    return triggers[change_name]
