import sys
import numpy as np

sys.path.insert(0, ".")
from qlora_mem.measure import measure_footprint
from qlora_mem.nf4 import compare_nf4_uniform
from qlora_mem.double import double_quant_size

def test_measure_footprint_valid():
    t = np.random.randn(128, 128).astype(np.float32)
    res = measure_footprint(t)
    assert res["quant_bytes"] < res["orig_bytes"]

def test_nf4_beats_uniform():
    t = np.random.randn(128, 128).astype(np.float32)
    res = compare_nf4_uniform(t)
    assert res["nf4_beats_uniform"] is True

def test_double_quant_compresses_absmax():
    t = np.random.randn(256, 256).astype(np.float32)
    res = double_quant_size(t)
    assert res["double_absmax_bytes"] < res["standard_absmax_bytes"]
