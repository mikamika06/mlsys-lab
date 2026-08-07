import struct
import numpy as np

GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q4_K_M = 12

def compute_tensor_bytes(n_elements, qtype):
    if qtype == GGML_TYPE_F16:
        return n_elements * 2
    elif qtype == GGML_TYPE_Q8_0:
        block_size = 32
        block_bytes = 34
        blocks = (n_elements + block_size - 1) // block_size
        return blocks * block_bytes
    elif qtype == GGML_TYPE_Q4_K or qtype == GGML_TYPE_Q4_K_M:
        block_size = 256
        block_bytes = 144
        blocks = (n_elements + block_size - 1) // block_size
        return blocks * block_bytes
    else:
        raise ValueError(f"Unknown type {qtype}")

def generate_mock_gguf():
    magic = b"GGUF"
    version = 3
    tensor_count = 2
    kv_count = 0
    header = struct.pack("<IQQQ", struct.unpack("<I", magic)[0], version, tensor_count, kv_count)

    tensors = [
        {"name": "token_embd.weight", "n_dims": 2, "dims": [512, 64], "type": GGML_TYPE_F16, "offset": 0},
        {"name": "output.weight", "n_dims": 2, "dims": [512, 64], "type": GGML_TYPE_Q8_0, "offset": 0}
    ]

    body = bytearray()
    body.extend(header)

    tensor_infos = []
    current_offset = 0

    for t in tensors:
        name_bytes = t["name"].encode("utf-8")
        name_len = len(name_bytes)
        t_info = bytearray()
        t_info.extend(struct.pack("<Q", name_len))
        t_info.extend(name_bytes)
        t_info.extend(struct.pack("<I", t["n_dims"]))
        for d in t["dims"]:
            t_info.extend(struct.pack("<Q", d))
        t_info.extend(struct.pack("<I", t["type"]))
        t_info.extend(struct.pack("<Q", current_offset))
        tensor_infos.append(t_info)

        n_el = int(np.prod(t["dims"]))
        current_offset += compute_tensor_bytes(n_el, t["type"])

    for t_info in tensor_infos:
        body.extend(t_info)

    return bytes(body), tensors
