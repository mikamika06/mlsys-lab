CONFIGS = [
    {"torch_version": "2.4.0", "cuda_version": "12.4", "flash_version": "2.6.3", "gpu_arch": "89"},
    {"torch_version": "2.5.0", "cuda_version": "12.6", "flash_version": "4.0.0", "gpu_arch": "90"},
    {"torch_version": "2.3.1", "cuda_version": "12.1", "flash_version": "2.5.8", "gpu_arch": "80"}
]

def build_dockerfile(cfg):
    lines = [
        f"FROM pytorch/pytorch:{cfg['torch_version']}-cuda{cfg['cuda_version']}-devel-ubuntu22.04",
        "ENV TORCH_CUDA_ARCH_LIST=" + cfg['gpu_arch'],
        "RUN pip install --no-cache-dir packaging ninja",
        f"RUN pip install --no-cache-dir flash-attn=={cfg['flash_version']} --no-build-isolation"
    ]
    return "\n".join(lines)

def compute_install_cost(cfg):
    base = 140.0 if cfg['flash_version'].startswith("4") else 90.0
    arch_mult = 1.25 if cfg['gpu_arch'] == "90" else 1.0
    return base * arch_mult

def compute_latency_ratio(cfg_a, cfg_b):
    return compute_install_cost(cfg_a) / (compute_install_cost(cfg_b) + 1e-6)
