import zlib


def compute_chkhsh(tokens: list[str], pre_tokenizer: str) -> int:
    data = pre_tokenizer.encode("utf-8") + b"\x00"
    for tok in tokens:
        tb = tok.encode("utf-8")
        data += len(tb).to_bytes(4, "little") + tb
    return zlib.crc32(data) & 0xFFFFFFFF
