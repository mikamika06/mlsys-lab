import numpy as np
from metal_op.model import FusedModel


def test_model_output_shape():
    model = FusedModel()
    x = np.ones((16, 16), dtype=np.float32)
    out = model.forward(x)
    assert out.shape == x.shape


def test_model_numerical_stability():
    model = FusedModel()
    x = np.zeros((8, 8), dtype=np.float32)
    out = model.forward(x)
    assert not np.isnan(out).any()
