import struct

MAGIC = b"TRT\x00"


def parse_plan_header(data: bytes) -> dict:
    if len(data) < 16:
        raise ValueError("Header too short")
    magic, trt_maj, trt_min, sm_maj, sm_min, flags, plen = struct.unpack(
        "<4sBBBBII", data[:16]
    )
    if magic != MAGIC:
        raise ValueError("Invalid magic bytes")
    if len(data) < 16 + plen:
        raise ValueError("Truncated engine binary")
    return {
        "magic": magic,
        "trt_version": (trt_maj, trt_min),
        "sm_version": (sm_maj, sm_min),
        "flags": flags,
        "payload_length": plen,
        "total_length": 16 + plen,
    }
