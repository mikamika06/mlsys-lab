import sys
import json

sys.path.insert(0, ".")
from qindex.writer import build_shard_index, serialize_index
from qindex.validation import validate_index_structure

SAMPLE_CP = [
    {
        "shard_name": "model-00001-of-00001.safetensors",
        "tensors": {
            "layer.0.weight": {"dtype": "q4_0", "shape": [128, 128], "data_offsets": [0, 8192]}
        }
    }
]

def test_index_structure():
    idx = build_shard_index(SAMPLE_CP)
    ser = serialize_index(idx)
    assert validate_index_structure(ser) is True

def test_offset_bounds_inversion():
    bad_idx = {
        "weight_map": {
            "layer.0.weight": {
                "file": "model.safetensors",
                "dtype": "q4_0",
                "shape": [128, 128],
                "offsets": [1000, 500]
            }
        },
        "metadata": {"total_size": 1000, "format": "quantized_v1"}
    }
    ser = json.dumps(bad_idx)
    assert validate_index_structure(ser) is False

def test_missing_metadata_fields():
    bad_idx = {
        "weight_map": {}
    }
    ser = json.dumps(bad_idx)
    assert validate_index_structure(ser) is False
