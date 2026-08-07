def resolve_shared_object(cuda_version: str, platform_tag: str) -> str:
    major = cuda_version.split(".")[0]
    if platform_tag == "linux_x86_64":
        return f"libbitsandbytes_cuda{major}.so"
    elif platform_tag == "linux_aarch64":
        return f"libbitsandbytes_cuda{major}_sbs.so"
    else:
        return f"libbitsandbytes_cpu.so"
