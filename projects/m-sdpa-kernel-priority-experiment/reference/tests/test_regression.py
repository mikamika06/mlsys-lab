import sys

sys.path.insert(0, ".")
from sdpa_exp.kernels import get_kernel_priority
from sdpa_exp.memory import compute_memory_blowup
from sdpa_exp.decoder import can_use_flash_attention

CONFIG = {"batch_size": 2, "seq_len": 4096, "num_heads": 8, "head_dim": 64, "dtype_bytes": 4}

def test_kernel_priority_valid():
    p = get_kernel_priority(CONFIG)
    assert isinstance(p, list) and len(p) > 0

def test_memory_blowup_positive():
    b = compute_memory_blowup(CONFIG)
    assert b > 1.0

def test_decoder_rejects_long_fp32():
    cfg = {"batch_size": 1, "seq_len": 4096, "num_heads": 8, "head_dim": 64, "dtype_bytes": 4}
    assert can_use_flash_attention(cfg) is False
