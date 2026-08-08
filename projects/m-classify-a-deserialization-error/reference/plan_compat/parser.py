import struct

MAGIC = b"TRT\x00"
OS_MAP = {1: "linux", 2: "windows", 3: "qnx"}
ARCH_MAP = {1: "x86_64", 2: "aarch64"}


def parse_header(data: bytes) -> dict:
    if len(data) < 32:
        return {
            "magic": b"",
            "trt_version": (0, 0, 0, 0),
            "sm": (0, 0),
            "os": "unknown",
            "arch": "unknown",
            "hardware_compatible": False,
            "lean_runtime": False,
            "payload_size": 0,
            "valid_checksum": False,
        }
    (
        magic,
        v_maj,
        v_min,
        v_pat,
        v_bld,
        sm_maj,
        sm_min,
        os_id,
        arch_id,
        flags,
        payload_sz,
        reserved,
        crc,
    ) = struct.unpack("!4sBBBBBBBBIQII", data[:32])
    calc_crc = sum(data[:28]) & 0xFFFFFFFF
    return {
        "magic": magic,
        "trt_version": (v_maj, v_min, v_pat, v_bld),
        "sm": (sm_maj, sm_min),
        "os": OS_MAP.get(os_id, "unknown"),
        "arch": ARCH_MAP.get(arch_id, "unknown"),
        "hardware_compatible": bool(flags & 0x01),
        "lean_runtime": bool(flags & 0x02),
        "payload_size": payload_sz,
        "valid_checksum": (calc_crc == crc) and (magic == MAGIC),
    }
