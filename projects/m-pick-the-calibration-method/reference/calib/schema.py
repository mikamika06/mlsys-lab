def select_quant_schema(min_val, max_val, has_zero_point_support=True):
    is_strictly_nonnegative = min_val >= 0.0
    symmetric_range = abs(min_val + max_val) < 0.1 * max(abs(min_val), abs(max_val), 1e-6)

    if is_strictly_nonnegative and has_zero_point_support:
        return {"schema": "U8S8", "activation_type": "uint8", "weight_type": "int8", "symmetric": False}
    elif symmetric_range or not has_zero_point_support:
        return {"schema": "S8S8", "activation_type": "int8", "weight_type": "int8", "symmetric": True}
    else:
        return {"schema": "U8S8", "activation_type": "uint8", "weight_type": "int8", "symmetric": False}
