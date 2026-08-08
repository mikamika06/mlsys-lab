import sys

sys.path.insert(0, ".")
from memacc.accounting import layer_eager_memory, layer_sdpa_memory


def test_sdpa_retained_memory_is_linear():
    cfg1 = {"batch_size": 1, "seq_len": 1024, "num_heads": 12, "head_dim": 64, "dtype": "float16"}
    cfg2 = {"batch_size": 1, "seq_len": 2048, "num_heads": 12, "head_dim": 64, "dtype": "float16"}

    mem1 = layer_sdpa_memory(cfg1)["retained_bytes"]
    mem2 = layer_sdpa_memory(cfg2)["retained_bytes"]

    ratio = mem2 / mem1
    assert abs(ratio - 2.0) < 0.05, f"SDPA retained memory scaling ratio should be 2.0, got {ratio}"


def test_eager_retained_memory_is_quadratic():
    cfg1 = {"batch_size": 1, "seq_len": 1024, "num_heads": 12, "head_dim": 64, "dtype": "float16"}
    cfg2 = {"batch_size": 1, "seq_len": 2048, "num_heads": 12, "head_dim": 64, "dtype": "float16"}

    mem1 = layer_eager_memory(cfg1)["retained_bytes"]
    mem2 = layer_eager_memory(cfg2)["retained_bytes"]

    ratio = mem2 / mem1
    assert ratio > 3.0, f"Eager retained memory should show quadratic growth, got ratio {ratio}"
