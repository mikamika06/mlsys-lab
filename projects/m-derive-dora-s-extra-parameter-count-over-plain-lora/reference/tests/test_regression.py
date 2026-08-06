import numpy as np
from adapter.scaling import rslora_scaling, plain_lora_scaling


def test_rslora_scaling_differs_from_plain():
    r = 64
    alpha = 16.0
    plain = plain_lora_scaling(r, alpha)
    rslora = rslora_scaling(r, alpha)
    assert not np.isclose(plain, rslora)


def test_rslora_scaling_behavior():
    r = 100
    alpha = 10.0
    got = rslora_scaling(r, alpha)
    want = alpha / np.sqrt(r)
    assert np.isclose(got, want)
