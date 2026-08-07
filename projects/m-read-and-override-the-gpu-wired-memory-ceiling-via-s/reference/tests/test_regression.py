import sys
import pytest

sys.path.insert(0, ".")
from sysctl_mem.ceiling import generate_sysctl_override, predict_wired_limit_mb


def test_override_safety():
    memsize = 64 * 1024 * 1024 * 1024
    cmd = generate_sysctl_override(memsize, 85.0)
    assert "iogpu.wired_mem_limit_mb=55705" in cmd

    try:
        generate_sysctl_override(memsize, 99.0)
        assert False, "Should have raised ValueError for percentage > 95"
    except ValueError:
        pass

    try:
        generate_sysctl_override(memsize, 40.0)
        assert False, "Should have raised ValueError for percentage < 50"
    except ValueError:
        pass


def test_prediction_scaling():
    limit_16 = predict_wired_limit_mb(16 * 1024 * 1024 * 1024)
    limit_128 = predict_wired_limit_mb(128 * 1024 * 1024 * 1024)
    assert limit_16 < limit_128
