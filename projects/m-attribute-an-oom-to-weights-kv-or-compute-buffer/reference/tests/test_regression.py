import memplan.oom

def test_regression():
    assert memplan.oom.attribute_oom(10, 8, 3, 1, False) == "kv"
