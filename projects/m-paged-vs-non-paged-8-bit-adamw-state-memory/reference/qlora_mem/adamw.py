import math

PAGE_TABLE_OVERHEAD_BYTES = 1048576


def compute_adamw_state_bytes(num_params: int, block_size: int = 256, paged: bool = False, max_layer_params: int = 0) -> int:
    if num_params <= 0:
        return 0
    blocks = math.ceil(num_params / block_size)
    non_paged_bytes = num_params * 2 + blocks * 8
    if not paged:
        return non_paged_bytes
    target_layer_params = max_layer_params if max_layer_params > 0 else num_params
    layer_blocks = math.ceil(target_layer_params / block_size)
    layer_bytes = target_layer_params * 2 + layer_blocks * 8
    paged_vram = layer_bytes + PAGE_TABLE_OVERHEAD_BYTES
    return min(non_paged_bytes, paged_vram)
