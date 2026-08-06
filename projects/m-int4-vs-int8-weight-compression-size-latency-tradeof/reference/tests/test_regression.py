import sys
sys.path.insert(0, ".")
from compressor.validator import validate_group_size
from compressor.tradeoff import compute_compressed_size, evaluate_tradeoff

def test_validate_group_size_divisible():
    assert validate_group_size(1024, 128) is True

def test_validate_group_size_raises_on_non_divisible():
    caught = False
    try:
        validate_group_size(1000, 128)
    except ValueError:
        caught = True
    assert caught, "Non-divisible group size did not raise ValueError"

def test_compressed_size_relation():
    s4 = compute_compressed_size(1024, 2048, 4)
    s8 = compute_compressed_size(1024, 2048, 8)
    assert s4 < s8, "INT4 compressed size should be smaller than INT8"
