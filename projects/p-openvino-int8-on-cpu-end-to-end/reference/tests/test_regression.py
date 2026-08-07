import sys
sys.path.insert(0, ".")
import numpy as np
from ov_engine.converter import convert_model
from ov_engine.profiler import profile_model
from ov_engine.quantizer import quantize_int8
from ov_engine.runtime import run_inference

def test_conversion_creates_file(tmp_path="."):
    out = "tmp_model.xml"
    convert_model("dummy", out)
    import os
    assert os.path.exists(out)
    if os.path.exists(out):
        os.remove(out)

def test_profiler_returns_dictionary():
    res = profile_model("dummy", np.zeros((1, 16)))
    assert isinstance(res, dict)
    assert len(res) > 0

def test_quantization_output():
    out = "tmp_int8.xml"
    quantize_int8("dummy", [np.zeros((1, 16))], out)
    import os
    assert os.path.exists(out)
    if os.path.exists(out):
        os.remove(out)

def test_runtime_latency_and_shape():
    inp = np.zeros((1, 16), dtype=np.float32)
    out = run_inference("dummy", inp, threads=4, latency_hint=True)
    assert out.shape == (1, 10)
