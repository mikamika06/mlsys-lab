import numpy as np
from benchaudit.stats import required_sample_size


def test_sample_size_regression():
    """Test sample size calculation against baseline expectations."""
    np.random.seed(42)
    samples = np.random.lognormal(mean=2.5, sigma=0.5, size=2000)
    res = required_sample_size(samples, target_ci_pct=0.02, confidence=0.95)
    assert res > 5000, f"Expected required sample size > 5000, got {res}"
    assert isinstance(res, int)
