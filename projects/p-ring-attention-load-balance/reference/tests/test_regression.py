import sys
sys.path.insert(0, ".")
from ring_attn.balancer import RingBalancer

def test_scaling_prediction():
    b = RingBalancer(4, 64)
    t1 = b.predict_scaling(4)
    t2 = b.predict_scaling(8)
    assert t1 > 0
    assert t2 > 0
    assert t1 != t2
