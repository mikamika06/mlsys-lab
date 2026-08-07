import sys

sys.path.insert(0, ".")
from opset.report import check_support
from opset.drift import migrate_squeeze_11, infer_resize_shape


def test_check_support_finds_unsupported():
    nodes = [{"name": "n1", "op_type": "Squeeze", "inputs": [], "outputs": [], "attributes": {}}]
    assert check_support(nodes, 12, {"Squeeze": 11}) == ["n1"]
    assert check_support(nodes, 10, {"Squeeze": 11}) == []


def test_migrate_squeeze_topology():
    nodes = [
        {"name": "n1", "op_type": "Squeeze", "inputs": ["X"], "outputs": ["Y"], "attributes": {"axes": [0]}}
    ]
    migrated = migrate_squeeze_11(nodes)
    assert len(migrated) == 2
    seen = {"X"}
    for n in migrated:
        for i in n["inputs"]:
            assert i in seen
        for o in n["outputs"]:
            seen.add(o)


def test_infer_resize_shape_prefers_sizes():
    shape = [10, 10]
    scales = [2.0, 2.0]
    sizes = [30, 30]
    assert infer_resize_shape(shape, scales, sizes, 11) == [30, 30]
    assert infer_resize_shape(shape, scales, sizes, 10) == [20, 20]
