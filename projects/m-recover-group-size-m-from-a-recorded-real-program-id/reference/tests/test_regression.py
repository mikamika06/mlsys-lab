import sys
sys.path.insert(0, ".")
from grouptrace.parser import parse_trace
from grouptrace.recover import recover_group_size

def test_parse_dimensions():
    trace = [(0, 0), (1, 0), (0, 1), (1, 1)]
    gm, gn = parse_trace(trace)
    assert gm == 2
    assert gn == 2

def test_recover_known_group_size():
    trace = [(0, 0), (1, 0), (0, 1), (1, 1)]
    g = recover_group_size(trace)
    assert g == 2
