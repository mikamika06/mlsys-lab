import struct


def parse_pte_constants(pte_bytes: bytes) -> dict:
    if len(pte_bytes) < 12:
        raise ValueError("Invalid PTE header size")
    magic, version, num_segments = struct.unpack("<4sII", pte_bytes[:12])
    if magic != b"PTE1" or version != 1:
        raise ValueError("Invalid magic or version")
    ptr = 12
    for _ in range(num_segments):
        if ptr + 24 > len(pte_bytes):
            raise ValueError("Truncated segment table")
        seg_type, offset, size, align = struct.unpack("<IQQI", pte_bytes[ptr : ptr + 24])
        ptr += 24
        if seg_type == 1:
            return {"offset": int(offset), "size": int(size), "alignment": int(align)}
    raise ValueError("Constant segment not found")
