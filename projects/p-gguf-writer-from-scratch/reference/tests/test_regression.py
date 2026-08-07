import os
import numpy as np
from gguf_writer.writer import GGUFWriter
from gguf_writer.tensor_map import map_tensor_name
from gguf_writer.validator import validate_gguf

def test_gguf_roundtrip(tmp_path=None):
    path = "test_model.gguf"
    writer = GGUFWriter(path)
    writer.add_metadata("general.architecture", "custom")
    writer.add_tensor("model.embed_tokens.weight", np.ones((16, 16), dtype=np.float32))
    writer.write()
    assert validate_gguf(path) is True
    if os.path.exists(path):
        os.remove(path)

def test_tensor_mapping():
    assert map_tensor_name("model.embed_tokens.weight") == "token_embd.weight"
    assert map_tensor_name("unknown.weight") == "unknown.weight"
