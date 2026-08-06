import numpy as np
from quantizer.params import calc_affine_params


def test_unclamped_zero_point_positive_range():
    scale, zp = calc_affine_params(10.0, 20.0, 0, 255)
    assert 0 <= zp <= 255


def test_unclamped_zero_point_negative_range():
    scale, zp = calc_affine_params(-50.0, -10.0, 0, 255)
    assert 0 <= zp <= 255
