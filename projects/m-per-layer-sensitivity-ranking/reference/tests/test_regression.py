import sys

sys.path.insert(0, ".")
from quant.allocation import allocate_bits
from quant.groups import emit_config_groups
from quant.sensitivity import compute_sensitivities

CONFIG = {
    "layers": [
        {"index": 0, "size": 1024},
        {"index": 1, "size": 1024},
        {"index": 2, "size": 1024}
    ]
}
STATS = {
    0: {"variance": 1.0, "mean": 0.5},
    1: {"variance": 4.0, "mean": 2.0},
    2: {"variance": 0.1, "mean": 0.1}
}


def test_allocation_respects_budget():
    sens = compute_sensitivities(CONFIG, STATS)
    bits = allocate_bits(sens, [2, 4, 8], 12)
    assert sum(bits) <= 12


def test_every_layer_has_assignment():
    sens = compute_sensitivities(CONFIG, STATS)
    bits = allocate_bits(sens, [2, 4, 8], 12)
    groups = emit_config_groups(CONFIG, bits)
    total_layers = sum(len(g["layers"]) for g in groups)
    assert total_layers == len(CONFIG["layers"])
