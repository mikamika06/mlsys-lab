import struct
import numpy as np

class GGUFWriter:
    def __init__(self, path):
        self.path = path
        self.metadata = {}
        self.tensors = []

    def add_metadata(self, key, value):
        self.metadata[key] = value

    def add_tensor(self, name, tensor):
        self.tensors.append((name, np.ascontiguousarray(tensor)))

    def write(self):
        with open(self.path, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))

            for k, v in self.metadata.items():
                k_bytes = k.encode("utf-8")
                f.write(struct.pack("<Q", len(k_bytes)))
                f.write(k_bytes)
                if isinstance(v, str):
                    f.write(struct.pack("<I", 8))
                    v_bytes = v.encode("utf-8")
                    f.write(struct.pack("<Q", len(v_bytes)))
                    f.write(v_bytes)
                elif isinstance(v, int):
                    f.write(struct.pack("<I", 5))
                    f.write(struct.pack("<q", v))
                elif isinstance(v, float):
                    f.write(struct.pack("<I", 6))
                    f.write(struct.pack("<d", v))

            tensor_infos = []
            current_offset = 0

            for name, tensor in self.tensors:
                n_bytes = name.encode("utf-8")
                dims = tensor.shape
                n_dims = len(dims)
                dtype_map = {np.float32: 0, np.float16: 1, np.int32: 2}
                dt = dtype_map.get(tensor.dtype.type, 0)
                nbytes = tensor.nbytes

                tensor_infos.append({
                    "name": n_bytes,
                    "dims": dims,
                    "dtype": dt,
                    "offset": current_offset,
                    "data": tensor.tobytes()
                })

                padded_size = (nbytes + 31) & ~31
                current_offset += padded_size

            for info in tensor_infos:
                f.write(struct.pack("<Q", len(info["name"])))
                f.write(info["name"])
                f.write(struct.pack("<I", len(info["dims"])))
                for d in info["dims"]:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I", info["dtype"]))
                f.write(struct.pack("<Q", info["offset"]))

            for info in tensor_infos:
                f.write(info["data"])
                pad = (31 - ((len(info["data"]) - 1) & 31)) if len(info["data"]) > 0 else 0
                if pad > 0 and pad < 32:
                    f.write(b"\x00" * pad)
