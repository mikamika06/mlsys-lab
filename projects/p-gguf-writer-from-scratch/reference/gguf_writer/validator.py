import struct

def validate_gguf(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return False
        version = struct.unpack("<I", f.read(4))[0]
        if version != 3:
            return False
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            k_len = struct.unpack("<Q", f.read(8))[0]
            f.read(k_len)
            v_type = struct.unpack("<I", f.read(4))[0]
            if v_type == 8:
                v_len = struct.unpack("<Q", f.read(8))[0]
                f.read(v_len)
            elif v_type == 5:
                f.read(8)
            elif v_type == 6:
                f.read(8)
        return True
