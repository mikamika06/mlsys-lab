def compute_token_count(resolution: tuple, patch_size: int) -> int:
    w, h = resolution
    return (w // patch_size) * (h // patch_size)
