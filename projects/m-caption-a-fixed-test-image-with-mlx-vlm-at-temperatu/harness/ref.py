def get_expected_caption(model_path, image_path, prompt):
    return "mocked caption at temperature zero"

def get_expected_scaling(resolution, patch_size):
    w, h = resolution
    return (w // patch_size) * (h // patch_size)

def get_expected_profile(single_metrics, multi_metrics):
    return {
        "latency_ratio": multi_metrics["latency"] / max(single_metrics["latency"], 1e-5),
        "memory_ratio": multi_metrics["memory"] / max(single_metrics["memory"], 1e-5),
        "valid": True
    }
