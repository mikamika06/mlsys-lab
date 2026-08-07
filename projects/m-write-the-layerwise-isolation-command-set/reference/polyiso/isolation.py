def build_isolation_command(onnx_path, trt_path, layer_names, output_json):
    cmd = [
        "polygraphy", "run", onnx_path,
        "--onnxrt",
        "--trt", trt_path,
        "--onnx-outputs"
    ]
    cmd.extend(layer_names)
    cmd.extend([
        "--trt-outputs"
    ])
    cmd.extend(layer_names)
    cmd.extend([
        "--save-inspect-iter-info", output_json
    ])
    return cmd


def build_mark_all_command(model_path, output_model_path):
    return [
        "polygraphy", "surgeon", "sanitize", model_path,
        "--override-outputs", "mark", "all",
        "-o", output_model_path
    ]
