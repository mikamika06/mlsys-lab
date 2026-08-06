import sys

sys.path.insert(0, ".")
from flashdiag.validator import validate_config


def test_validator_rejects_invalid_head_dim():
    assert validate_config({"head_dim": 48}) is False


def test_validator_accepts_valid_head_dim():
    assert validate_config({"head_dim": 64}) is True


def test_validator_handles_empty_config():
    assert validate_config({}) is False
