import sys
sys.path.insert(0, ".")
from kvmem.solver import predict_num_gpu_blocks, solve_max_model_len

def test_max_len_is_aligned_to_block_size():
    config = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8
    }

    total_vram = 24 * 1024**3
    weights_size = 15 * 1024**3
    util = 0.90
    block_size = 16
    dtype = "float16"

    max_len = solve_max_model_len(config, dtype, total_vram, weights_size, util, block_size)
    blocks = predict_num_gpu_blocks(config, dtype, total_vram, weights_size, util, block_size)

    assert max_len == blocks * block_size, "Max model length must be exactly blocks * block_size"
    assert max_len % block_size == 0, "Max model len is not block-aligned"


def test_no_vram_returns_zero():
    config = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8
    }
    max_len = solve_max_model_len(config, "float16", 1024**3, 10 * 1024**3, 0.9, 16)
    assert max_len == 0, "Should return 0 length when no VRAM is available"
