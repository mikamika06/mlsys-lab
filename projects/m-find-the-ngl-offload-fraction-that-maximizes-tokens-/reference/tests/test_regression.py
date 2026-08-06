import sys
sys.path.insert(0, ".")
from metalopt.offload import find_optimal_ngl, calculate_throughput_ratio
from metalopt.quant import check_tensor_divisible_by_256, compare_quantization_profiles


def test_offload_bounds():
    layers = 32
    vram_limit = 4096
    layer_mem = 200
    ngl, tps = find_optimal_ngl(layers, vram_limit, layer_mem)
    assert ngl * layer_mem <= vram_limit
    assert ngl >= 0


def test_tensor_dimension_check():
    try:
        check_tensor_divisible_by_256((512, 127))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not divisible by 256" in str(e)
