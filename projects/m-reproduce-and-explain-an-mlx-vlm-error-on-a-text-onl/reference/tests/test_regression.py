import sys
sys.path.insert(0, ".")
from mlx_vlm_edge.token_counter import compute_image_tokens

def test_token_counter_basic():
    assert compute_image_tokens(336, 14) == 576

def test_token_counter_merged():
    cfg = {"spatial_merge_size": 2}
    assert compute_image_tokens(336, 14, vision_config=cfg) == 144

def test_token_counter_invariant():
    res1 = compute_image_tokens(224, 16)
    res2 = compute_image_tokens(448, 16)
    assert res2 == res1 * 4
