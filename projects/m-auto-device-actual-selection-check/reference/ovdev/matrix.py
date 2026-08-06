def check_compile_support(device_spec, shape_spec):
    """Check if model shape configuration is supported on given target device."""
    dev_name = device_spec.get("name", "")
    supports_dynamic = device_spec.get("supports_dynamic", False)
    max_dims = device_spec.get("max_dims", 4)
    
    is_dynamic = shape_spec.get("is_dynamic", False)
    rank = len(shape_spec.get("dims", []))

    if rank > max_dims:
        return False, f"Rank {rank} exceeds device max {max_dims}"

    if is_dynamic and not supports_dynamic:
        return False, f"Device {dev_name} does not support dynamic shapes"

    if not is_dynamic:
        for d in shape_spec.get("dims", []):
            if d <= 0:
                return False, "Invalid non-positive static dimension"

    return True, "OK"


def build_compile_matrix(devices, model_shapes):
    """Build a matrix mapping (device, shape) pairs to compilation success status."""
    matrix = {}
    for dev in devices:
        dev_name = dev["name"]
        matrix[dev_name] = {}
        for shape in model_shapes:
            shape_name = shape["name"]
            success, msg = check_compile_support(dev, shape)
            matrix[dev_name][shape_name] = {
                "success": success,
                "reason": msg
            }
    return matrix
