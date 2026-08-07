import sys

sys.path.insert(0, ".")
from tp_marlin.analyze import pad_for_marlin, check_eligibility


def test_padding_is_strictly_minimal():
    layers = [{"name": "l1", "style": "row", "in_features": 129, "out_features": 257}]
    tp_size = 2
    padded = pad_for_marlin(layers, tp_size)
    assert padded[0]["in_features"] == 256
    assert padded[0]["out_features"] == 512


def test_no_padding_when_already_aligned():
    layers = [{"name": "l2", "style": "col", "in_features": 128, "out_features": 512}]
    tp_size = 2
    padded = pad_for_marlin(layers, tp_size)
    assert padded[0]["in_features"] == 128
    assert padded[0]["out_features"] == 512


def test_padded_is_eligible():
    layers = [{"name": "l3", "style": "row", "in_features": 100, "out_features": 100}]
    tp_size = 4
    padded = pad_for_marlin(layers, tp_size)
    assert check_eligibility(padded, tp_size)[0] is True
