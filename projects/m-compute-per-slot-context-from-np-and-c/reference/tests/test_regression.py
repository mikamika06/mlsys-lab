from slotplan.metrics import compute_cache_reuse_ratio


def test_cache_reuse_ratio_calculation():
    """Test calculation of cache reuse ratio from metrics."""
    sample_metrics = (
        "# HELP llamacpp:prompt_tokens_processed_total Total prompt tokens\n"
        "llamacpp:prompt_tokens_processed_total 300\n"
        "# HELP llamacpp:prompt_tokens_cached_total Total cached prompt tokens\n"
        "llamacpp:prompt_tokens_cached_total 700\n"
    )
    ratio = compute_cache_reuse_ratio(sample_metrics)
    assert abs(ratio - 0.7) < 1e-6
