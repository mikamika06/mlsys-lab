import torch

def detect_stack():
    cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    fa_ver = None
    try:
        import flash_attn
        fa_ver = getattr(flash_attn, "__version__", None)
    except ImportError:
        pass
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "flash_attn_version": fa_ver
    }
