import sys

sys.path.insert(0, ".")
from tensorsplit.sizes import compute_layer_sizes
from tensorsplit.split import compute_tensor_split

CONFIG = {
    "layers": [
        {"index": 0, "type": "embd", "hidden_dim": 4096, "ffn_dim": 11008},
        {"index": 1, "type": "attn", "hidden_dim": 4096, "kv_heads": 8, "head_dim": 128},
        {"index": 2, "type": "attn", "hidden_dim": 4096, "kv_heads": 8, "head_dim": 128},
        {"index": 3, "type": "output", "hidden_dim": 4096, "vocab_size": 32000}
    ],
    "bytes_per_param": 2
}


def test_split_sums_to_one():
    split = compute_tensor_split(CONFIG)
    assert len(split) == 2
    assert abs(sum(split) - 1.0) < 1e-6


def test_split_fractions_valid():
    split = compute_tensor_split(CONFIG)
    assert 0.0 < split[0] < 1.0
    assert 0.0 < split[1] < 1.0


def test_layer_sizes_positive():
    sizes = compute_layer_sizes(CONFIG)
    assert all(s > 0 for s in sizes)
