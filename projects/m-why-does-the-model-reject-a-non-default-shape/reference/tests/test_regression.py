import sys
sys.path.insert(0, ".")
from edgeexport.shapes import validate_shape, evaluate_enumeration
from edgeexport.checker import check_symbolic_propagation
from edgeexport.compiler import compile_with_shapes


def test_validate_shape_exact():
    assert validate_shape([1, 64, 128], [1, 64, 128]) is True
    assert validate_shape([1, 64, 128], [1, 32, 128]) is False


def test_evaluate_enumeration():
    shapes = [[1, 32], [1, 64], [1, 128]]
    constraints = [{"dim": 1, "min": 50}]
    res = evaluate_enumeration(shapes, constraints)
    assert len(res) == 2


def test_compile_budget():
    res = compile_with_shapes(None, [[1, 32]], 100)
    assert res["status"] == "compiled"
