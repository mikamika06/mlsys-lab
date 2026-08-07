import struct


def extract_gguf_template(metadata_bytes):
    if b"tokenizer.chat_template" in metadata_bytes:
        idx = metadata_bytes.find(b"tokenizer.chat_template")
        val_start = idx + len(b"tokenizer.chat_template")
        if val_start + 4 <= len(metadata_bytes):
            length = struct.unpack("<I", metadata_bytes[val_start:val_start+4])[0]
            start_str = val_start + 4
            return metadata_bytes[start_str:start_str+length].decode("utf-8", errors="ignore")
    return ""
