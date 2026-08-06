import sys
import numpy as np

sys.path.insert(0, ".")
from ggufmap.mapper import map_hf_to_gguf
from ggufmap.rope import undo_rope_permutation
from ggufmap.filter import match_tensors


def test_map_hf_to_gguf():
    assert map_hf_to_gguf("model.embed_tokens.weight") == "token_embd.weight"
    assert map_hf_to_gguf("model.layers.0.self_attn.q_proj.weight") == "blk.0.attn_q.weight"
    assert map_hf_to_gguf("model.layers.3.mlp.down_proj.bias") == "blk.3.ffn_down.bias"


def test_undo_rope_permutation():
    arr = np.arange(16, dtype=np.float32)
    res = undo_rope_permutation(arr, n_heads=2)
    expected = np.array([0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15], dtype=np.float32)
    assert np.array_equal(res, expected)
    assert not np.array_equal(res, arr)


def test_match_tensors():
    names = ["blk.0.attn_q.weight", "blk.0.ffn_gate.weight", "output.weight"]
    matched = match_tensors(names, r"attn_q")
    assert matched == ["blk.0.attn_q.weight"]
