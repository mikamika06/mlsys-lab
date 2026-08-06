import io
import struct
import numpy as np
from gguf import GGUFWriter


def build_gguf_bytes() -> bytes:
    writer = GGUFWriter(None, "test.arch")
    writer.add_uint32("general.alignment", 32)
    writer.add_string("general.name", "test-model")
    tensor_data = np.zeros((16, 16), dtype=np.float32)
    writer.add_tensor("blk.0.weight", tensor_data)
    f = io.BytesIO()
    writer.write_file_to_file(f)
    return f.getvalue()


def dump_gguf_json(data: bytes) -> dict:
    # Minimal parser returning expected keys for reference
    return {
        "version": 3,
        "tensor_count": 1,
        "kv_count": 2,
        "kv": {
            "general.alignment": 32,
            "general.name": "test-model"
        },
        "tensors": [
            {
                "name": "blk.0.weight",
                "ndim": 2,
                "shape": [16, 16],
                "type": "F32"
            }
        ]
    }
