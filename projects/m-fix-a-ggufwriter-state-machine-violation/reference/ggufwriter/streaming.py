import os
import struct

class GGUFTensorStreamer:
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        if not os.path.exists(self.filepath):
            return
        with open(self.filepath, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return
            num_tensors = struct.unpack("<I", f.read(4))[0]
            for i in range(num_tensors):
                yield f"tensor_{i}", i * 1024
