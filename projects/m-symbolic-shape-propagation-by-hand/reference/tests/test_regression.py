from symshape.infer import propagate_shapes


def test_symbolic_shape_invariants():
    graph = {
        "inputs": {
            "A": ("batch", 16, 32),
            "B": (1, 16, 32)
        },
        "nodes": [
            {
                "name": "add1",
                "op": "Add",
                "inputs": ["A", "B"],
                "params": {}
            }
        ]
    }
    res = propagate_shapes(graph)
    assert res["add1"] == ("batch", 16, 32)

    invalid_graph = {
        "inputs": {
            "A": ("batch", 16, 32),
            "B": ("batch", 8, 32)
        },
        "nodes": [
            {
                "name": "add1",
                "op": "Add",
                "inputs": ["A", "B"],
                "params": {}
            }
        ]
    }
    res_inv = propagate_shapes(invalid_graph)
    assert "add1" not in res_inv or res_inv["add1"] is None
