def assert_symbolic_axes(onnx_model, batch_dim_name="batch"):
    inputs = onnx_model.get("input", [])
    for inp in inputs:
        shape = inp.get("shape", [])
        if shape and shape[0] != batch_dim_name:
            raise AssertionError(f"Expected dynamic axis {batch_dim_name}, got {shape[0]}")
    return True
