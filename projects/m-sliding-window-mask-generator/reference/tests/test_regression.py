import sys

sys.path.insert(0, ".")
from swm.masking import generate_sliding_window_mask
from swm.memory import kv_cache_memory_bytes


def test_mask_strictly_obeys_sliding_window():
    mask = generate_sliding_window_mask(10, 3)
    assert not mask[9, 5], "Mask allows attention completely outside the sliding window"
    assert mask[9, 8], "Mask improperly blocks attention immediately preceding the token"
    assert not mask[2, 3], "Mask allows future tokens (violates causal invariant)"


def test_memory_plateaus_for_sliding_layers():
    layers = [
        {"type": "global"},
        {"type": "sliding", "window_size": 100}
    ]
    mem_small = kv_cache_memory_bytes(50, 1, layers, 8, 128)
    mem_large = kv_cache_memory_bytes(500, 1, layers, 8, 128)

    bytes_per_token = 2 * 8 * 128 * 2
    assert mem_small == 100 * bytes_per_token, "Incorrect footprint for seq_len < window_size"
    assert mem_large == 600 * bytes_per_token, "Footprint did not plateau for sliding layer"
