def get_sample_metadata():
    return {"cuda_version": 122, "tensorrt_version": 102, "gpu_compute_capability": 89}

def get_sample_profiles():
    return [{"min_shape": [1, 64], "opt_shape": [4, 64], "max_shape": [8, 64]}]

def get_sample_engines():
    ea = {"tactics": [101, 102, 103], "weights_hash": "hash_v1"}
    eb = {"tactics": [101, 102, 103], "weights_hash": "hash_v1"}
    return ea, eb
