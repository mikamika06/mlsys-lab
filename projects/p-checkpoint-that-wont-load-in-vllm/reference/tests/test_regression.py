import sys
sys.path.insert(0, ".")
import numpy as np
from qfix.parser import parse_checkpoint
from qfix.metadata import map_metadata
from qfix.packing import fix_packing
from qfix.verify import verify_output
from qfix.engine import load_in_engine

def test_parser_validates_dict():
    data = {"layers.0.weight": [1, 2]}
    parsed = parse_checkpoint(data)
    assert "layers.0.weight" in parsed

def test_metadata_mapping():
    data = {"layers.0.weight": [1, 2]}
    parsed = parse_checkpoint(data)
    mapped = map_metadata(parsed)
    assert "model.layers.0.weight" in mapped

def test_packing_behavior():
    data = {"layers.0.weight": np.array([0x21], dtype=np.uint8)}
    packed = fix_packing(data)
    assert "layers.0.weight" in packed

def test_engine_execution():
    data = {"layers.0.weight": np.array([0x21], dtype=np.uint8)}
    inputs = np.array([1, 1], dtype=np.float32)
    out = load_in_engine(data, inputs)
    assert isinstance(out, float)
