import sys

sys.path.insert(0, ".")
from shapes.verifier import Dim, resolve_module_signature


def test_auto_rejects_varying_shapes():
    dims = [Dim.auto("batch")]
    try:
        resolve_module_signature(dims, [(32,), (64,)])
        assert False, "AUTO should reject varying shapes"
    except ValueError:
        pass


def test_dynamic_accepts_varying_shapes():
    dims = [Dim.dynamic("seq")]
    res = resolve_module_signature(dims, [(128,), (512,)])
    assert res[0].min_val == 128
    assert res[0].max_val == 512
