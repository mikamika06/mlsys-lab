import sys
sys.path.insert(0, ".")
from ane_model.audit import get_block_placement, find_fallback_ops
from ane_model.transform import make_ane_friendly, verify_parity, measure_ane_fraction, measure_energy_per_request
import numpy as np

def test_placement_audit():
    model = {"blocks": [{"device": "GPU", "fallback_op": "reshape"}]}
    p = get_block_placement(model)
    assert len(p) == 1

def test_fallback_detection():
    model = {"blocks": [{"device": "GPU", "fallback_op": "reshape"}]}
    ops = find_fallback_ops(model)
    assert "reshape" in ops

def test_transformation_and_parity():
    model = {"blocks": [{"device": "GPU", "fallback_op": "reshape"}]}
    inp = np.ones((1, 10))
    t_model = make_ane_friendly(model)
    diff = verify_parity(model, t_model, inp)
    assert diff < 0.001

def test_ane_fraction_high():
    model = {"blocks": [{"device": "GPU", "fallback_op": "reshape"}]}
    t_model = make_ane_friendly(model)
    frac = measure_ane_fraction(t_model, np.ones((1, 10)))
    assert frac >= 0.9

def test_energy_improvement():
    model = {"blocks": [{"device": "GPU", "fallback_op": "reshape"}]}
    t_model = make_ane_friendly(model)
    e_before = measure_energy_per_request(model, np.ones((1, 10)))
    e_after = measure_energy_per_request(t_model, np.ones((1, 10)))
    assert e_after < e_before
