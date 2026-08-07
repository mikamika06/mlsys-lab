import struct

MAGIC = b"TRTTC"


def parse_header(binary_data: bytes) -> dict:
    if len(binary_data) < 24:
        raise ValueError("Binary header too short")

    magic = binary_data[:5]
    if magic != MAGIC:
        raise ValueError("Invalid magic bytes")

    format_version, trt_major, trt_minor, trt_patch, sm_major, sm_minor, sm_count, tactic_sources, entry_count = struct.unpack(
        "<BBBBBBHII", binary_data[5:24]
    )

    return {
        "magic": magic.decode("ascii"),
        "format_version": format_version,
        "trt_version": (trt_major, trt_minor, trt_patch),
        "sm_version": (sm_major, sm_minor),
        "sm_count": sm_count,
        "tactic_sources": tactic_sources,
        "entry_count": entry_count,
        "header_size": 24,
    }


def parse_entries(binary_data: bytes) -> list:
    header = parse_header(binary_data)
    offset = header["header_size"]
    entries = []

    for _ in range(header["entry_count"]):
        if offset + 20 > len(binary_data):
            raise ValueError("Truncated payload")
        op_id, tactic_id, tactic_source, latency_us = struct.unpack(
            "<IIId", binary_data[offset : offset + 20]
        )
        entries.append(
            {
                "op_id": op_id,
                "tactic_id": tactic_id,
                "tactic_source": tactic_source,
                "latency_us": float(latency_us),
            }
        )
        offset += 20

    return entries
