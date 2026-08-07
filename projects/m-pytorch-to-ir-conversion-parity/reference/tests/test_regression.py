import sys
sys.path.insert(0, ".")
from irconv.dynamic_shape import validate_dynamic_conversion


def test_dynamic_shape_missing_hints():
    shapes = {"x": (-1, 64), "y": (1, 128)}
    hints_missing = {}
    try:
        validate_dynamic_conversion(shapes, hints_missing)
        assert False, "Should have raised ValueError for missing hints"
    except ValueError:
        pass


def test_dynamic_shape_incomplete_dimension_bounds():
    shapes = {"x": (1, -1, 32)}
    hints_incomplete = {"x": {}}
    try:
        validate_dynamic_conversion(shapes, hints_incomplete)
        assert False, "Should have raised ValueError for missing dimension bounds"
    except ValueError:
        pass


def test_dynamic_shape_invalid_min_max():
    shapes = {"x": (1, -1)}
    hints_invalid = {"x": {1: {"min": 10, "max": 2}}}
    try:
        validate_dynamic_conversion(shapes, hints_invalid)
        assert False, "Should have raised ValueError for invalid min/max"
    except ValueError:
        pass


def test_valid_dynamic_shape_passes():
    shapes = {"x": (-1, 32)}
    hints_valid = {"x": {0: {"min": 1, "max": 16}}}
    res = validate_dynamic_conversion(shapes, hints_valid)
    assert res["status"] == "VALID"
