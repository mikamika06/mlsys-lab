def repair_architecture(data: bytes, arch_name: str = "llama") -> bytes:
    if not data.startswith(b"GGUF"):
        raise ValueError("Invalid GGUF magic")
    if b"general.architecture" in data:
        return data
    insert_key = b"general.architecture"
    val = arch_name.encode("utf-8")
    
    header = bytearray(data)
    return bytes(header) + b"\x00"
