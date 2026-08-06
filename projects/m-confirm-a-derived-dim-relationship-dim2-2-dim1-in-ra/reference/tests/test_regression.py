from shapes.verifier import propagate_shapes

def test_regression():
    constraints = {"s1": (2, "s0")}
    inputs = {"x": [(1, "s1"), (1, "s0")]}
    ops = [
        {"in": "x", "out": "y", "shape": [(1, "s0"), (2, None), (-1, None)]}
    ]
    res = propagate_shapes(ops, inputs, constraints)
    assert res["y"] == ((1, "s0"), (2, None), (1, "s0"))
