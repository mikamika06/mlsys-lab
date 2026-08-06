def detect_shadowing_wheels(distributions_list):
    ort_packages = []
    for dist in distributions_list:
        name = dist.get("name", "").lower()
        if "onnxruntime" in name:
            ort_packages.append(dist)
    has_cpu = any("onnxruntime" == p.get("name", "").lower() for p in ort_packages)
    has_gpu = any("onnxruntime-gpu" == p.get("name", "").lower() for p in ort_packages)
    shadowed = has_cpu and has_gpu
    return {
        "packages": ort_packages,
        "shadowed": shadowed
    }
