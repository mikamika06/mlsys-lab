import sys
import numpy as np
sys.path.insert(0, ".")
from dcpfix.parser import parse_dcp_metadata
from dcpfix.stitch import stitch_shards


def test_metadata_parsing_structure():
    meta = {
        "storage_data": {
            "weight": {
                "shape": [4, 4],
                "offsets": [[0, 0], [2, 0]],
                "lengths": [[2, 4], [2, 4]],
                "file_name": ["shard_0.pt", "shard_1.pt"]
            }
        }
    }
    parsed = parse_dcp_metadata(meta)
    assert "weight" in parsed
    assert parsed["weight"]["shape"] == [4, 4]
    assert len(parsed["weight"]["offsets"]) == 2


def test_stitch_correctness():
    meta = {
        "storage_data": {
            "layer.weight": {
                "shape": [2, 2],
                "offsets": [[0, 0], [0, 1]],
                "lengths": [[2, 1], [2, 1]],
                "file_name": ["s0.bin", "s1.bin"]
            }
        }
    }
    s0 = np.array([1.0, 2.0], dtype=np.float32).tobytes()
    s1 = np.array([3.0, 4.0], dtype=np.float32).tobytes()
    shards = {"s0.bin": s0, "s1.bin": s1}

    stitched = stitch_shards(shards, parse_dcp_metadata(meta))
    assert "layer.weight" in stitched
    expected = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    np.testing.assert_array_equal(stitched["layer.weight"], expected)
