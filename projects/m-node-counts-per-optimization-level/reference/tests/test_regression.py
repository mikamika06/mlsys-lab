import sys
sys.path.insert(0, ".")
from ortopt.fusions import check_portability


def test_portability_constraints():
    model = {"layout": "NHWC", "target": "strict_nchw_only"}
    assert check_portability(model) is False
    good_model = {"layout": "NCHW", "target": "strict_nchw_only"}
    assert check_portability(good_model) is True
