"""Learner regression test suite implementation."""

import sys
sys.path.insert(0, ".")

from exporter.budget import compute_kv_bytes, compute_max_context
from exporter.convert import build_toy_decoder_spec, simulate_export
from exporter.repair import check_state_alignment, repair_state_names


def test_state_name_repair():
    """Verify that state name mismatch is correctly detected and repaired."""
    spec = build_toy_decoder_spec(
        num_layers=2, hidden_dim=128, num_kv_heads=4, head_dim=32, max_context=512
    )
    mismatched_export = simulate_export(
        spec, state_names=("mismatched_k", "mismatched_v")
    )
    expected = ("key_cache", "value_cache")

    assert not check_state_alignment(mismatched_export, expected)
    repaired = repair_state_names(mismatched_export, expected)
    assert check_state_alignment(repaired, expected)


def test_kv_cache_budget_bounds():
    """Verify that computed max context strictly respects byte limits."""
    num_layers = 12
    num_kv_heads = 8
    head_dim = 64
    budget = 10 * 1024 * 1024
    dtype = "float16"

    ctx = compute_max_context(num_layers, num_kv_heads, head_dim, budget, dtype)
    used_bytes = compute_kv_bytes(num_layers, num_kv_heads, head_dim, ctx, dtype)
    over_bytes = compute_kv_bytes(num_layers, num_kv_heads, head_dim, ctx + 1, dtype)

    assert used_bytes <= budget
    assert over_bytes > budget
