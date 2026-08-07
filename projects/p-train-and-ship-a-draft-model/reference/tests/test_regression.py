import sys
import numpy as np

sys.path.insert(0, ".")
from spec.draft import DraftModel
from spec.quant import QuantizedDraft

def test_memory_budget():
    d = DraftModel(32, 8)
    qd = QuantizedDraft(d)

    assert qd.w1_q.dtype == np.int8
    assert qd.w2_q.dtype == np.int8

    size = qd.w1_q.nbytes + qd.w2_q.nbytes
    budget = 32 * 8 * 2

    assert size <= budget
