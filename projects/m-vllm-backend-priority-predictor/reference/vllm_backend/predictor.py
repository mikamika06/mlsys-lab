PRIORITY = ["FLASHINFER", "FLASH_ATTN", "XFORMERS", "TORCH_SDPA", "ROCM_FLASH"]


def predict_backend(config, candidate_backends):
    rejections = {}
    ordered_candidates = [b for b in PRIORITY if b in candidate_backends]
    for b in candidate_backends:
        if b not in ordered_candidates:
            ordered_candidates.append(b)

    device = config.get("device", "cuda")
    sm_version = config.get("sm_version", 80)
    head_dim = config.get("head_dim", 128)
    dtype = config.get("dtype", "float16")
    is_causal = config.get("is_causal", True)
    sliding_window = config.get("sliding_window", None)

    for backend in ordered_candidates:
        if backend == "FLASHINFER":
            if device != "cuda":
                rejections[backend] = "Requires CUDA device"
                continue
            if sm_version < 80:
                rejections[backend] = "Requires compute capability >= 8.0"
                continue
            if head_dim not in (64, 128, 256):
                rejections[backend] = f"Unsupported head dimension {head_dim}"
                continue
            return backend, rejections

        elif backend == "FLASH_ATTN":
            if device != "cuda":
                rejections[backend] = "Requires CUDA device"
                continue
            if sm_version < 80:
                rejections[backend] = "Requires compute capability >= 8.0"
                continue
            if dtype not in ("float16", "bfloat16"):
                rejections[backend] = f"Unsupported dtype {dtype}"
                continue
            return backend, rejections

        elif backend == "ROCM_FLASH":
            if device != "rocm":
                rejections[backend] = "Requires ROCm device"
                continue
            return backend, rejections

        elif backend == "XFORMERS":
            if device not in ("cuda", "cpu"):
                rejections[backend] = "Unsupported device platform"
                continue
            if sliding_window is not None:
                rejections[backend] = "Sliding window attention not supported"
                continue
            return backend, rejections

        elif backend == "TORCH_SDPA":
            return backend, rejections

        else:
            rejections[backend] = "Unknown backend requested"

    return None, rejections
