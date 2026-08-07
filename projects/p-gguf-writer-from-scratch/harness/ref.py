import os
import numpy as np
from gguf_writer.writer import GGUFWriter
from gguf_writer.tensor_map import map_tensor_name
from gguf_writer.validator import validate_gguf

def generate_fixture(path):
    writer = GGUFWriter(path)
    writer.add_metadata("general.architecture", "custom")
    writer.add_tensor("model.embed_tokens.weight", np.zeros((4, 4), dtype=np.float32))
    writer.write()
    return path
