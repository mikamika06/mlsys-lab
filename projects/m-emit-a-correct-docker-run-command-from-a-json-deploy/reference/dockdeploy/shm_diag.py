def _parse_size_bytes(size_str: str) -> int:
    size_str = str(size_str).strip().upper()
    if size_str.endswith("G") or size_str.endswith("GB"):
        num = float(size_str.rstrip("GB"))
        return int(num * 1024 * 1024 * 1024)
    if size_str.endswith("M") or size_str.endswith("MB"):
        num = float(size_str.rstrip("MB"))
        return int(num * 1024 * 1024)
    if size_str.endswith("K") or size_str.endswith("KB"):
        num = float(size_str.rstrip("KB"))
        return int(num * 1024)
    return int(size_str)


def diagnose_shm_failure(spec: dict) -> dict:
    gpus = int(spec.get("gpus", 1))
    tp = int(spec.get("tensor_parallel_size", gpus))

    nccl_per_gpu = 1024 * 1024 * 1024
    torch_shm_baseline = 2 * 1024 * 1024 * 1024

    required_bytes = (tp * nccl_per_gpu) + torch_shm_baseline

    ipc_mode = spec.get("ipc", "")
    shm_size_str = spec.get("shm_size", "64m")

    if ipc_mode == "host":
        current_bytes = 64 * 1024 * 1024 * 1024
    else:
        current_bytes = _parse_size_bytes(shm_size_str)

    is_sufficient = current_bytes >= required_bytes
    required_gb = int((required_bytes + (1024**3 - 1)) // (1024**3))

    return {
        "required_bytes": required_bytes,
        "current_bytes": current_bytes,
        "is_sufficient": is_sufficient,
        "recommended_flag": f"--shm-size={required_gb}g" if not is_sufficient else ""
    }
