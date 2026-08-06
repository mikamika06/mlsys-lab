def diagnose_failure(logs: str, spec: dict) -> dict:
    issues = []
    shm_size = spec.get("shm_size")
    ipc = spec.get("ipc")
    gpus = spec.get("gpus", 0)

    if isinstance(gpus, int) and gpus > 1 and ipc != "host":
        if not shm_size:

            issues.append("shm_too_small")
        else:

            pass

    if "RuntimeError: NCCL error" in logs or "torch.distributed" in logs or "Bus error" in logs:
        if not shm_size and ipc != "host":
            issues.append("shm_too_small")

    if "CUDA out of memory" in logs:
        issues.append("cuda_oom")

    if not issues and ("segmentation fault" in logs.lower() or "sigsegv" in logs.lower()):
        issues.append("shm_too_small")

    return {
        "issues": list(set(issues)),
        "is_multi_gpu": bool(isinstance(gpus, int) and gpus > 1 or isinstance(gpus, str) and gpus != "0"),
        "ipc_mode": ipc,
        "shm_size": shm_size
    }
