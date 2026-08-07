import sys
sys.path.insert(0, ".")
from det.trainer import is_deterministic

def test_toggle_flag():
    assert is_deterministic() is True
