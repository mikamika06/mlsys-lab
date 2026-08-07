def build_isolation_command(onnx_path, trt_path, layer_names, output_json):
    """Build the polygraphy run command line for layerwise isolation."""
    raise NotImplementedError


def build_mark_all_command(model_path, output_model_path):
    """Build polygraphy surgeon command to mark all intermediate tensors as outputs."""
    raise NotImplementedError
