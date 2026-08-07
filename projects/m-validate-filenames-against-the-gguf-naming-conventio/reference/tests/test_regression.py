import sys
sys.path.insert(0, ".")
from gguf_tool.validate import validate_filename
from gguf_tool.roundtrip import gguf_to_torch, torch_to_gguf
from gguf_tool.diff import diff_metadata
import torch

def test_validate_filename_valid():
    assert validate_filename("llama-3-8b-instruct-q4_k_m.gguf") is True
    assert validate_filename("model-f16.gguf") is True

def test_validate_filename_invalid():
    assert validate_filename("model.gguf") is False
    assert validate_filename("llama-3-8b-q4_k_m.bin") is False

def test_roundtrip_consistency():
    meta = {"architecture": "llama", "context_length": 4096}
    tensors = {"blk.0.attn_q.weight": torch.zeros((16, 16))}
    gguf_obj = torch_to_gguf(tensors, meta)
    recovered = gguf_to_torch(gguf_obj)
    assert "blk.0.attn_q.weight" in recovered
    assert recovered["blk.0.attn_q.weight"].shape == (16, 16)

def test_diff_metadata_detects_mismatch():
    hub = {"version": 1, "arch": "llama"}
    local = {"version": 2, "arch": "llama"}
    d = diff_metadata(hub, local)
    assert "version" in d
    assert "arch" not in d
