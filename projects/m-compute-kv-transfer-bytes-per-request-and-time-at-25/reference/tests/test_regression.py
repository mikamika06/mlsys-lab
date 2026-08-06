from kvtransfer.transfer import compute_kv_bytes, compute_transfer_times
from kvtransfer.breakeven import compute_breakeven_prompt_len
from kvtransfer.sizing import size_prefill_decode_ratio

def test_transfer_bytes_positive():
    cfg = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2}
    b = compute_kv_bytes(cfg, 100)
    assert b > 0
    times = compute_transfer_times(b, [100.0])
    assert 100.0 in times
    assert times[100.0] > 0

def test_breakeven_positive():
    cfg = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2}
    pl = compute_breakeven_prompt_len(cfg, 100.0)
    assert pl >= 1

def test_sizing_ratio():
    ratio = size_prefill_decode_ratio(1000, 100)
    assert ratio > 0
