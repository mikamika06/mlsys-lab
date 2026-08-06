import sys
sys.path.insert(0, ".")
from hlodiff.compile import validate_hlo_compilation


def test_good_hlo_compiles():
    assert validate_hlo_compilation("HloModule valid") is True


def test_bad_hlo_fails():
    try:
        validate_hlo_compilation("HloModule mismatch")
        assert False, "should have failed"
    except RuntimeError:
        pass
