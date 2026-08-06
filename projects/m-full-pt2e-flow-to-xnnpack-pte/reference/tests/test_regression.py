import sys

sys.path.insert(0, ".")
from pt2ex.quant import observe_ranges, compute_qparams, convert_tensor
from pt2ex.export import export_pte
import numpy as np


def test_observe_ranges_valid():
    acts = [np.array([-1.0, 2.0], dtype=np.float32)]
    res = observe_ranges(acts)
    assert res["min"][0] == -1.0
    assert res["max"][0] == 2.0


def test_qparams_per_channel():
    w = np.random.randn(10, 10).astype(np.float32)
    qp = compute_qparams(w, per_channel=True, axis=0)
    assert qp["per_channel"] is True
    assert len(qp["scale"]) == 10


def test_export_pte_output():
    d = {"weight": [1, 2, 3]}
    b = export_pte(d)
    assert isinstance(b, bytes)
    assert len(b) > 0
