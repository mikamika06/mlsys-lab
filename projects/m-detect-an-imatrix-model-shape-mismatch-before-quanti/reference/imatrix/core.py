def verify_tensor_shape(tensor_name, model_shape, imatrix_entry):
    if not isinstance(model_shape, (list, tuple)):
        raise TypeError("model shape must be a list or tuple")
    if "shape" not in imatrix_entry:
        raise KeyError("imatrix entry missing shape")
    expected_shape = imatrix_entry["shape"]
    if list(model_shape) != list(expected_shape):
        return False, f"shape mismatch for {tensor_name}: model {model_shape} vs imatrix {expected_shape}"
    return True, ""
