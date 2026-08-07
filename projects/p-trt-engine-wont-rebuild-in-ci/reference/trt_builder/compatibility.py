def check_compatibility(metadata):
    required = ["cuda_version", "tensorrt_version", "gpu_compute_capability"]
    for r in required:
        if r not in metadata:
            return False
    return metadata["cuda_version"] >= 120 and metadata["tensorrt_version"] >= 100
