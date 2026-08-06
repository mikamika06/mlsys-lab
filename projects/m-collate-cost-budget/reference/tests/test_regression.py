"""Regression tests for collation cost budgeting."""

from collate.budget import profile_collate, evaluate_budget


def test_collate_budget_compliance():
    """Verify collation budget checks correctly enforce limits."""
    def mock_sample_gen(bs):
        return [i for i in range(bs)]

    def mock_collate(samples):
        bs = len(samples)
        simulated_time_sec = bs * 0.001
        return 0.0, simulated_time_sec

    profile = profile_collate(mock_collate, mock_sample_gen, [10, 20])
    eval_res = evaluate_budget(profile, max_budget_ms_per_batch=15.0, target_throughput_samples_sec=500.0)

    assert eval_res[10]["compliant"] is True
    assert eval_res[20]["within_budget"] is False
    assert eval_res[20]["compliant"] is False
