def detect_mismatch(config_text, filenames):
    backend = None
    platform = None
    for line in config_text.splitlines():
        line = line.strip()
        if line.startswith("backend:"):
            backend = line.split(":")[1].strip().strip('"')
        elif line.startswith("platform:"):
            platform = line.split(":")[1].strip().strip('"')

    has_onnx = any(f.endswith(".onnx") for f in filenames)
    has_plan = any(f.endswith(".plan") or f.endswith(".trt") for f in filenames)
    has_savedmodel = any("savedmodel" in f for f in filenames)
    has_pt = any(f.endswith(".pt") or f.endswith(".pth") for f in filenames)

    if backend == "onnxruntime" and not has_onnx:
        return True
    if platform == "tensorflow_graphdef" and not has_savedmodel:
        return True
    if platform == "tensorrt_plan" and not has_plan:
        return True
    if backend == "pytorch" and not has_pt:
        return True
    if not backend and not platform:
        return True
    return False
