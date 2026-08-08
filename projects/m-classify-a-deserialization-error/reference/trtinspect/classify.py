import struct

MAGIC = b"TRT\x00"


def classify_deserialization_error(data: bytes, target_trt_version: tuple) -> str:
    if len(data) < 16:
        return "CORRUPT_HEADER"
    magic, trt_maj, trt_min, sm_maj, sm_min, flags, plen = struct.unpack(
        "<4sBBBBII", data[:16]
    )
    if magic != MAGIC:
        return "BAD_MAGIC"
    if len(data) < 16 + plen:
        return "TRUNCATED_PAYLOAD"
    if (trt_maj, trt_min) != target_trt_version:
        return "TRT_VERSION_MISMATCH"
    if flags != 0:
        return "UNSUPPORTED_FLAGS"
    return "OK"
