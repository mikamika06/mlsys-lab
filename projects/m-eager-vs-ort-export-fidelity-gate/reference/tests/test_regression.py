import sys

sys.path.insert(0, ".")
from exportgate.constants import detect_baked_constants


def test_detect_baked_constants_finds_constants():
    model = {"nodes": [{"op": "Constant", "val": 256, "is_baked": True}]}
    assert detect_baked_constants(model) > 0, "failed to detect baked constant"


def test_detect_baked_constants_ignores_parameters():
    model = {"nodes": [{"op": "Param", "name": "seq_len"}]}
    assert detect_baked_constants(model) == 0, "falsely flagged non-baked parameter"
