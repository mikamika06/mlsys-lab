import sys

sys.path.insert(0, ".")
from troubleshoot.preflight import can_fit


def test_preflight_basic_fit():
    assert can_fit(7, 2048, 2, 80.0) is True


def test_preflight_excessive_model():
    assert can_fit(1000, 2048, 1, 80.0) is False
