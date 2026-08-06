def configure_hopper_fa2(device_name: str, compute_capability: tuple[int, int], env: dict[str, str]) -> dict[str, str]:
    """Force FlashAttention-2 backend selection on Hopper architecture."""
    major, minor = compute_capability
    is_hopper = major == 9 or "H100" in device_name.upper() or "H800" in device_name.upper()
    if not is_hopper:
        raise ValueError(f"Device {device_name} with capability {compute_capability} is not a Hopper node.")
    updated_env = dict(env)
    updated_env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    updated_env["FLASH_ATTENTION_FORCE_BUILD"] = "1"
    return updated_env


def select_backend(device_name: str, compute_capability: tuple[int, int], force_fa2: bool, env: dict[str, str]) -> str:
    """Select appropriate attention backend given hardware and constraints."""
    if force_fa2:
        configure_hopper_fa2(device_name, compute_capability, env)
        return "FLASH_ATTN"
    
    env_backend = env.get("VLLM_ATTENTION_BACKEND")
    if env_backend:
        return env_backend

    major, _ = compute_capability
    if major >= 9:
        return "FLASH_ATTN"
    elif major >= 8:
        return "FLASHINFER"
    return "XFORMERS"
