import struct


def extract_chat_template(gguf_bytes: bytes) -> str:
    key = b"tokenizer.chat_template"
    idx = gguf_bytes.find(key)
    if idx == -1:
        return ""
    pos = idx + len(key)
    length = struct.unpack("<I", gguf_bytes[pos : pos + 4])[0]
    return gguf_bytes[pos + 4 : pos + 4 + length].decode("utf-8")
