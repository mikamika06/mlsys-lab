import struct
import numpy as np

def emit_gguf(metadata: dict, tensors: dict, path: str) -> None:
    """Emit a GGUF file matching reference structure."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(metadata)))

        for k, v in sorted(metadata.items()):
            kb = k.encode("utf-8")
            f.write(struct.pack("<Q", len(kb)))
            f.write(kb)
            if isinstance(v, int):
                f.write(struct.pack("<I", 0))
                f.write(struct.pack("<q", v))
            elif isinstance(v, str):
                f.write(struct.pack("<I", 8))
                vb = v.encode("utf-8")
                f.write(struct.pack("<Q", len(vb)))
                f.write(vb)
            elif isinstance(v, float):
                f.write(struct.pack("<I", 4))
                f.write(struct.pack("<d", v))

        tensor_data_list = []
        offset = 0
        for name, data in sorted(tensors.items()):
            nb = name.encode("utf-8")
            f.write(struct.pack("<Q", len(nb)))
            f.write(nb)
            shape = data.shape
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", 0))
            f.write(struct.pack("<Q", offset))
            raw = np.ascontiguousarray(data, dtype=np.float32).tobytes()
            tensor_data_list.append(raw)
            offset += len(raw)

        for raw in tensor_data_list:
            f.write(raw)
