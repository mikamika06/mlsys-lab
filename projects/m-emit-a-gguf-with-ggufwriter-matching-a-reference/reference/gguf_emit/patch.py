import struct
import numpy as np

def dump_gguf_json(path: str) -> dict:
    """Dump GGUF metadata and tensor info as JSON-compatible dict."""
    with open(path, "rb") as f:
        magic = f.read(4)
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        metadata = {}
        for _ in range(n_kv):
            k_len = struct.unpack("<Q", f.read(8))[0]
            k = f.read(k_len).decode("utf-8")
            v_type = struct.unpack("<I", f.read(4))[0]
            if v_type == 0:
                v = struct.unpack("<q", f.read(8))[0]
            elif v_type == 8:
                v_len = struct.unpack("<Q", f.read(8))[0]
                v = f.read(v_len).decode("utf-8")
            elif v_type == 4:
                v = struct.unpack("<d", f.read(8))[0]
            else:
                v = None
            metadata[k] = v

        tensors = []
        for _ in range(n_tensors):
            name_len = struct.unpack("<Q", f.read(8))[0]
            name = f.read(name_len).decode("utf-8")
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            t_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append({"name": name, "dimensions": dims, "type": t_type, "offset": offset})

        return {
            "magic": magic.decode("utf-8"),
            "version": version,
            "metadata": metadata,
            "tensors": tensors
        }

def patch_metadata(path: str, new_metadata: dict, out_path: str) -> None:
    """Patch metadata in place and preserve tensor bytes."""
    with open(path, "rb") as f:
        content = f.read()

    old_dump = dump_gguf_json(path)

    # Locate where tensor data starts by finding the first tensor offset or re-parsing header
    # For robust patching, let's parse header components precisely
    f_iter = 0
    magic = content[f_iter:f_iter+4]; f_iter += 4
    version = struct.unpack("<I", content[f_iter:f_iter+4])[0]; f_iter += 4
    n_tensors = struct.unpack("<Q", content[f_iter:f_iter+8])[0]; f_iter += 8

    # Skip old n_kv and old metadata to find tensor headers start
    old_n_kv = struct.unpack("<Q", content[f_iter:f_iter+8])[0]; f_iter += 8
    for _ in range(old_n_kv):
        k_len = struct.unpack("<Q", content[f_iter:f_iter+8])[0]; f_iter += 8
        f_iter += k_len
        v_type = struct.unpack("<I", content[f_iter:f_iter+4])[0]; f_iter += 4
        if v_type == 0:
            f_iter += 8
        elif v_type == 8:
            v_len = struct.unpack("<Q", content[f_iter:f_iter+8])[0]; f_iter += 8
            f_iter += v_len
        elif v_type == 4:
            f_iter += 8

    tensor_headers_start = f_iter

    # Find tensor data start by scanning through tensor headers
    t_iter = tensor_headers_start
    for _ in range(n_tensors):
        name_len = struct.unpack("<Q", content[t_iter:t_iter+8])[0]; t_iter += 8
        t_iter += name_len
        n_dims = struct.unpack("<I", content[t_iter:t_iter+4])[0]; t_iter += 4
        t_iter += n_dims * 8
        t_iter += 4 # type
        t_iter += 8 # offset
    tensor_data_start = t_iter
    tensor_bytes = content[tensor_data_start:]

    # Build new header with new metadata
    out_parts = [magic, struct.pack("<I", version), struct.pack("<Q", n_tensors), struct.pack("<Q", len(new_metadata))]
    for k, v in sorted(new_metadata.items()):
        kb = k.encode("utf-8")
        out_parts.append(struct.pack("<Q", len(kb)))
        out_parts.append(kb)
        if isinstance(v, int):
            out_parts.append(struct.pack("<I", 0))
            out_parts.append(struct.pack("<q", v))
        elif isinstance(v, str):
            out_parts.append(struct.pack("<I", 8))
            vb = v.encode("utf-8")
            out_parts.append(struct.pack("<Q", len(vb)))
            out_parts.append(vb)
        elif isinstance(v, float):
            out_parts.append(struct.pack("<I", 4))
            out_parts.append(struct.pack("<d", v))

    # Append original tensor headers section
    out_parts.append(content[tensor_headers_start:tensor_data_start])
    # Append tensor bytes
    out_parts.append(tensor_bytes)

    with open(out_path, "wb") as f:
        for p in out_parts:
            f.write(p)
