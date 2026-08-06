def generate_dockerfile(config):
    t = config.get("torch_version", "2.4.0")
    c = config.get("cuda_version", "12.4")
    f = config.get("flash_version", "2.6.3")
    g = config.get("gpu_arch", "89")
    return f"FROM pytorch/pytorch:{t}-cuda{c}-devel-ubuntu22.04\nENV TORCH_CUDA_ARCH_LIST={g}\nRUN pip install --no-cache-dir packaging ninja\nRUN pip install --no-cache-dir flash-attn=={f} --no-build-isolation"
