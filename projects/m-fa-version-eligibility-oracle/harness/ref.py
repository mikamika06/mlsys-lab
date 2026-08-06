import packaging.version

ENVIRONMENTS = [
    {
        "torch_version": "2.3.0+cu121",
        "cuda_version": "12.1",
        "device_name": "NVIDIA A100-SXM4-80GB",
        "compute_capability": (8, 0),
        "installed_packages": {"torch": "2.3.0", "flash_attn": "2.5.8"}
    },
    {
        "torch_version": "2.1.2+cu118",
        "cuda_version": "11.8",
        "device_name": "NVIDIA RTX 3090",
        "compute_capability": (8, 6),
        "installed_packages": {"torch": "2.1.2", "flash_attn": "1.0.9"}
    },
    {
        "torch_version": "2.4.0+cu124",
        "cuda_version": "12.4",
        "device_name": "NVIDIA H100-SXM5-80GB",
        "compute_capability": (9, 0),
        "installed_packages": {"torch": "2.4.0", "flash_attn": "2.6.3"}
    }
]

def detect_stack_from_env(env):
    cc = env["compute_capability"]
    major, minor = cc
    return {
        "torch_version": env["torch_version"],
        "cuda_version": env["cuda_version"],
        "compute_capability": f"{major}.{minor}",
        "device_name": env["device_name"],
        "flash_attn_version": env["installed_packages"].get("flash_attn")
    }

def evaluate_eligibility(stack):
    cc = tuple(map(int, stack["compute_capability"].split(".")))
    fa = stack.get("flash_attn_version")
    fa_ver = packaging.version.Version(fa) if fa else None

    eligible = {}
    eligible["fa1"] = cc >= (7, 0) and fa_ver is not None and fa_ver.major == 1
    eligible["fa2"] = cc >= (8, 0) and fa_ver is not None and fa_ver.major == 2 and fa_ver.minor >= 5
    eligible["fa3"] = cc >= (9, 0) and fa_ver is not None and fa_ver.major >= 2 and fa_ver.minor >= 6
    return eligible

def plan_upgrade(stack, target_version):
    cc = tuple(map(int, stack["compute_capability"].split(".")))
    if target_version == "fa3" and cc < (9, 0):
        return {"action": "hardware_upgrade", "target_gpu": "H100", "min_cuda": "12.4"}
    if target_version == "fa2" and cc < (8, 0):
        return {"action": "hardware_upgrade", "target_gpu": "A100", "min_cuda": "11.8"}
    fa = stack.get("flash_attn_version")
    fa_ver = packaging.version.Version(fa) if fa else None
    if target_version == "fa2" and (fa_ver is None or fa_ver.major < 2):
        return {"action": "pip_install", "package": "flash-attn>=2.5.8"}
    return {"action": "none"}
