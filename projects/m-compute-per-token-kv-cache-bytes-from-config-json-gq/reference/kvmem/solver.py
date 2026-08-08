from kvmem.config import get_block_size_bytes

def predict_num_gpu_blocks(config: dict, dtype: str, total_vram: int, weights_size: int, util: float, block_size: int) -> int:
    available = int(total_vram * util) - weights_size
    if available <= 0:
        return 0
    block_bytes = get_block_size_bytes(config, dtype, block_size)
    return available // block_bytes


def solve_max_model_len(config: dict, dtype: str, total_vram: int, weights_size: int, util: float, block_size: int) -> int:
    blocks = predict_num_gpu_blocks(config, dtype, total_vram, weights_size, util, block_size)
    return blocks * block_size
