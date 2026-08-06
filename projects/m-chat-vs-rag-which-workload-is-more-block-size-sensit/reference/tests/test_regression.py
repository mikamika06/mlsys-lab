"""Learner regression test suite."""

from block_sensitivity.workload import explain_hit_rate_regression


def test_hit_rate_regression():
    """Verify hit rate drops on unaligned prefix matching when block size doubles."""
    shared_prefix_len = 1050
    suffixes = [100, 120, 90, 110]

    res = explain_hit_rate_regression(
        shared_prefix_len=shared_prefix_len,
        request_suffix_lengths=suffixes,
        block_size_1=16,
        block_size_2=32
    )

    assert res["cached_tokens_bs1"] == 1040
    assert res["cached_tokens_bs2"] == 1024
    assert res["unaligned_tail_loss"] == 16
    assert res["hit_rate_drop"] > 0.0
