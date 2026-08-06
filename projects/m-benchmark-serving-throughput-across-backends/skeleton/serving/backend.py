def configure_hopper_fa2(device_name: str, compute_capability: tuple[int, int], env: dict[str, str]) -> dict[str, str]:
    """Force FlashAttention-2 backend selection on Hopper architecture."""
    raise NotImplementedError


def select_backend(device_name: str, compute_capability: tuple[int, int], force_fa2: bool, env: dict[str, str]) -> str:
    """Select appropriate attention backend given hardware and constraints."""
    raise NotImplementedError
