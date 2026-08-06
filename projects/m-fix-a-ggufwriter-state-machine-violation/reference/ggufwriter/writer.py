import os
import struct

class GGUFWriter:
    def __init__(self, path, endian="little"):
        self.path = path
        self.endian = endian
        self.state = 0
        self.kv_data = []
        self.tensors = []

    def add_header(self, kv):
        if self.state > 1:
            raise RuntimeError("State violation: header added after tensors")
        self.state = 1
        self.kv_data.append(kv)

    def add_tensor(self, name, data, tensor_type=None):
        if self.state == 0:
            self.state = 1
        self.tensors.append((name, data, tensor_type))

    def write_header_to_file(self):
        self.state = 2
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.kv_data)))
