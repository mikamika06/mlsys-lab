import sys
sys.path.insert(0, ".")
import numpy as np
from bnb_ledger.ledger import get_ledger, predict_memory_footprint
from bnb_ledger.quant import nested_absmax_quantize

def test_ledger_bits_positive():
    config = {"name": "test", "bits_per_weight": 4, "double_quant": True, "nested_bits": 8, "block_size": 256}
    ledger = get_ledger(config)
    assert ledger["bits_per_param"] > 0

def test_predict_memory_footprint():
    config = {"name": "test", "bits_per_weight": 4, "double_quant": False}
    mem = predict_memory_footprint(config, 1000)
    assert mem > 0

def test_nested_quant_shape():
    x = np.random.randn(512)
    out = nested_absmax_quantize(x, block_size=256)
    assert len(out) == len(x)
