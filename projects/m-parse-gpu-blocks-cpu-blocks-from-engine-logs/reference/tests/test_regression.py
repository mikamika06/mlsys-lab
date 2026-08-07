import sys
sys.path.insert(0, ".")
from kvparse.parser import parse_blocks
from kvparse.capacity import analyze_tp_scaling


def test_parser_basic():
    res = parse_blocks("INFO: # GPU blocks: 1000, # CPU blocks: 500")
    assert res["gpu_blocks"] == 1000
    assert res["cpu_blocks"] == 500


def test_scaling_basic():
    res = analyze_tp_scaling("# GPU blocks: 1000", "# GPU blocks: 2000")
    assert res["doubles_capacity"] is True
