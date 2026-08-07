def validate_backend(model_files, config_content):
    backend = "unknown"
    for line in config_content.splitlines():
        if "backend" in line:
            parts = line.split(":")
            if len(parts) > 1:
                backend = parts[1].strip().strip('"\'')

    if "onnx" in backend.lower() and not any(f.endswith(".onnx") for f in model_files):
        return False, "onnx backend specified but no .onnx file found"
    if "pytorch" in backend.lower() and not any(f.endswith(".pt") or f.endswith(".pth") for f in model_files):
        return False, "pytorch backend specified but no model weights found"
    return True, "ok"
