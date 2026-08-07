import sys
sys.path.insert(0, ".")
from roofline.analysis import find_decode_compute_bound_batch_size, calculate_operational_intensity

MODEL = {
    "num_params": 7000000000,
    "bytes_per_param": 2,
    "num_layers": 32,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "context_len": 2048
}

HARDWARE = {
    "peak_flops": 312e12,
    "peak_bandwidth": 2000e9
}


def test_decode_batch_transition():
    b = find_decode_compute_bound_batch_size(MODEL, HARDWARE)
    assert isinstance(b, int)
    assert b > 1


def test_operational_intensity_monotonicity():
    i1 = calculate_operational_intensity(MODEL, 1, 1024, phase="decode")
    i2 = calculate_operational_intensity(MODEL, 64, 1024, phase="decode")
    assert i2 > i1
