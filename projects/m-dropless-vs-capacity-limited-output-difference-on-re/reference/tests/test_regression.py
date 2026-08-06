import sys
import numpy as np

sys.path.insert(0, ".")
from moe_routing.capacity import select_capacity_factor

def test_select_capacity_factor_valid_range():
    tokens = np.zeros((16, 4))
    logits = np.zeros((16, 2))
    cf = select_capacity_factor(tokens, logits, top_k=1, max_drop_rate=0.0)
    assert cf >= 0.25 and cf <= 2.0

def test_select_capacity_factor_zero_tolerance():
    tokens = np.zeros((8, 2))
    logits = np.zeros((8, 2))
    cf = select_capacity_factor(tokens, logits, top_k=1, max_drop_rate=0.0)
    assert isinstance(cf, float) or isinstance(cf, int)
