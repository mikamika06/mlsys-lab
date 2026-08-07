import sys
import numpy as np

sys.path.insert(0, ".")
from aneattn.layout import to_ane_friendly
from aneattn.metrics import count_ops
from aneattn.convert import measure_latency


def test_layout_shape_and_dimensions():
    rng = np.random.default_rng(1337)
    x = rng.standard_normal((1, 4, 16, 32))
    out = to_ane_friendly(x)
    assert out.ndim == 4
    assert out.shape == (1, 128, 1, 16)


def test_metrics_consistency():
    naive_ops = count_ops("naive")
    ane_ops = count_ops("ane")
    assert ane_ops["reshape"] < naive_ops["reshape"]
    assert ane_ops["transpose"] < naive_ops["transpose"]


def test_latency_delta():
    naive_lat = measure_latency("naive")
    ane_lat = measure_latency("ane")
    assert ane_lat < naive_lat
