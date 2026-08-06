import pytest
from triton_matmul.constraints import check_tile_alignment


def test_valid_alignment():
    assert check_tile_alignment(32, 32, 32) is True


def test_invalid_alignment():
    assert check_tile_alignment(15, 32, 32) is False
