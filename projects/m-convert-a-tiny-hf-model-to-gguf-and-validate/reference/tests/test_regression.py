import numpy as np
from ggufconv.converter import SynthModelConverter, register_converter
from ggufconv.precision import convert_outtype

def test_conversion_pipeline():
    cfg = {"hidden_size": 32, "num_attention_heads": 4}
    conv = SynthModelConverter(cfg)
    tensors = {"base_model.weight": np.ones((32, 32), dtype=np.float32)}
    converted = conv.convert_tensors(tensors)
    assert "weight" in converted
    res = convert_outtype(converted, "f16")
    assert res["weight"].dtype == np.float16

def test_precision_bounds():
    w = {"w": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    res = convert_outtype(w, "f16")
    assert len(res) == 1
