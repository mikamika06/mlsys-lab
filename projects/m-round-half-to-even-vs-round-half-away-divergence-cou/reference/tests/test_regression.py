import sys
sys.path.insert(0, ".")
from quantexport.divergence import count_divergences
from quantexport.costs import requant_cost_share
from quantexport.detector import detect_wrong_dimension

def test_divergence_non_negative():
    arr = [0.1, 0.5, 1.5, 2.5]
    assert count_divergences(arr) >= 0

def test_cost_share_bounds():
    nodes = [{"type": "requantize", "cycles": 100}]
    share = requant_cost_share(nodes, 200)
    assert 0.0 <= share <= 1.0

def test_detector_valid():
    assert detect_wrong_dimension([16, 32], 0) is False
    assert detect_wrong_dimension([16, 32], 5) is True
