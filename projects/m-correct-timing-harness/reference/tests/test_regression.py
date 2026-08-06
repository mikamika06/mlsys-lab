from timing.stats import calculate_sample_size


def test_sample_size_scaling():
    n1 = calculate_sample_size(mean=10.0, std=1.0, target_error=0.05)
    n2 = calculate_sample_size(mean=10.0, std=2.0, target_error=0.05)
    assert n2 > n1, "sample size must increase with higher standard deviation"


def test_sample_size_bounds():
    n = calculate_sample_size(mean=5.0, std=0.5, target_error=0.05)
    assert isinstance(n, int)
    assert n > 0
