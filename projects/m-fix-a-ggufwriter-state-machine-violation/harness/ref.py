import os
import struct

def generate_test_file(path, num_tensors=2):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", num_tensors))
        f.write(struct.pack("<Q", 1))
    return path
