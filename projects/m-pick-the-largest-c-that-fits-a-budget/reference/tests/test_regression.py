import sys
sys.path.insert(0, ".")
from kvcache.budget import find_largest_context
from kvcache.flash import requires_flash_attention
from kvcache.asym import compute_fused_penalty


def test_budget_monotonicity():
    cfg = {"base_bytes": 1024, "bytes_per_token": 16}
    c1 = find_largest_context(cfg, 10000)
    c2 = find_largest_context(cfg, 20000)
    assert c1 <= c2


def test_flash_requirement():
    assert requires_flash_attention("Q4_0") is True


def test_asym_penalty():
    assert compute_fused_penalty("Q4_0", "Q8_0") > 1.0
