from recomp.guards import validate_guard

def test_guard_validation():
    assert validate_guard(10, 20, "shape") is True
    assert validate_guard(25, 20, "shape") is False
    assert validate_guard(20, 20, "value") is True
    assert validate_guard(21, 20, "value") is False
