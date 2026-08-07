import sys
import numpy as np
import struct

sys.path.insert(0, ".")
from interop.mapping import map_mlx_block_weights
from interop.diffing import diff_model_weights
from interop.safetensors import compute_safetensors_header


def test_map_mlx_block_weights_keys():
    weights = {
        "attention.wq.weight": np.ones((4, 4)),
        "feed_forward.w1.weight": np.ones((8, 4)),
    }
    mapped = map_mlx_block_weights(weights, block_idx=2)
    assert "model.layers.2.self_attn.q_proj.weight" in mapped
    assert "model.layers.2.mlp.gate_proj.weight" in mapped


def test_diff_model_weights_matching():
    g = {"blk.0.attn_q.weight": np.array([1.0, 2.0])}
    c = {"model.layers.0.self_attn.q_proj.weight": np.array([1.0, 2.0])}
    res = diff_model_weights(g, c)
    assert res["common_keys"] == 1
    assert res["matched_keys"] == 1
    assert res["max_abs_diff"] == 0.0


def test_safetensors_header_prefix_length():
    tensors = {"x": np.zeros((4,), dtype=np.float32)}
    length, prefix = compute_safetensors_header(tensors)
    assert len(prefix) == 8
    unpacked_len = struct.unpack("<Q", prefix)[0]
    assert unpacked_len == length
    expected_json = '{"x":{"data_offsets":[0,16],"dtype":"F32","shape":[4]}}'.encode("utf-8")
    assert length == len(expected_json)
