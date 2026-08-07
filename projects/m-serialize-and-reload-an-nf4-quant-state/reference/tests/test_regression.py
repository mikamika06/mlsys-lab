import sys
sys.path.insert(0, ".")
from quantutil.serialize import serialize_quant_state
from quantutil.reload import reload_quant_state
from quantutil.config import rebuild_quantization_config

def test_serialize_reload_roundtrip():
    class MockState:
        def __init__(self):
            self.absmax = [0.1, 0.2, 0.3]
            self.bits = 4
            self.quant_type = "nf4"
    orig = MockState()
    ser = serialize_quant_state(orig)
    reloaded = reload_quant_state(ser)
    assert reloaded.bits == 4
    assert reloaded.quant_type == "nf4"

def test_rebuild_config_from_metadata():
    meta = {"bits": "4", "bnb_4bit_quant_type": "nf4", "quant_method": "bitsandbytes"}
    cfg = rebuild_quantization_config(meta)
    assert cfg["bits"] == 4
    assert cfg["bnb_4bit_quant_type"] == "nf4"
