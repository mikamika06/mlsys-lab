import struct


def parse_mil_header(blob_bytes):
    if len(blob_bytes) < 16:
        raise ValueError("blob too short")
    magic, version, payload_offset, flags = struct.unpack("<4sIII", blob_bytes[:16])
    if magic != b"MIL1":
        raise ValueError("invalid magic")
    return {
        "magic": magic.decode("ascii"),
        "version": version,
        "payload_offset": payload_offset,
        "flags": flags,
    }
