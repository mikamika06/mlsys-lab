import packaging.version

def plan_upgrade(stack_info, target_version):
    cc_str = stack_info.get("compute_capability", "0.0")
    cc = tuple(map(int, cc_str.split(".")))
    if target_version == "fa3" and cc < (9, 0):
        return {"action": "hardware_upgrade", "target_gpu": "H100", "min_cuda": "12.4"}
    if target_version == "fa2" and cc < (8, 0):
        return {"action": "hardware_upgrade", "target_gpu": "A100", "min_cuda": "11.8"}
    fa = stack_info.get("flash_attn_version")
    fa_ver = packaging.version.Version(fa) if fa else None
    if target_version == "fa2" and (fa_ver is None or fa_ver.major < 2):
        return {"action": "pip_install", "package": "flash-attn>=2.5.8"}
    return {"action": "none"}
