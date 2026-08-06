def check_compile_support(device_spec, shape_spec):
    """Check if model shape configuration is supported on given target device."""
    raise NotImplementedError


def build_compile_matrix(devices, model_shapes):
    """Build a matrix mapping (device, shape) pairs to compilation success status."""
    raise NotImplementedError
