import relays.compare as c
import relays.fold as f
import relays.model as m

def test_model_structure():
    model = m.make_3op_model()
    assert len(model["ops"]) == 3

def test_ir_counts():
    model = m.make_3op_model()
    res = c.compare_ir_counts(model)
    assert res["relay_count"] > 0
    assert res["relax_count"] > 0

def test_folding_discrepancy():
    model = m.make_3op_model()
    res = f.check_constant_folding(model)
    assert res["discrepancy"] > 0
