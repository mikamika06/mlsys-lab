PAGE_TABLE_OVERHEAD_BYTES = 1048576


def compute_adamw_state_bytes(num_params: int, block_size: int = 256, paged: bool = False, max_layer_params: int = 0) -> int:
    raise NotImplementedError
