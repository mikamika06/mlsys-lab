import struct


def convert_dat_to_gguf(dat_path: str, gguf_path: str) -> None:
    with open(dat_path, "rb") as f:
        data = f.read()
    magic = b"GGUF"
    version = struct.pack("<I", 3)
    tensor_count = struct.pack("<Q", 1)
    kv_count = struct.pack("<Q", 0)
    header = magic + version + tensor_count + kv_count
    with open(gguf_path, "wb") as f:
        f.write(header)
        f.write(data)
