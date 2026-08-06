import struct

MAGIC = b"TRT\x01"
PLATFORMS = {
    1: "linux-x86_64",
    2: "linux-aarch64",
    3: "windows-x86_64"
}


def parse_plan_header(data: bytes) -> dict:
    """Parse raw plan header bytes into metadata dict."""
    if len(data) < 28:
        return {"valid": False}
    magic, v_maj, v_min, v_pat, v_bld, sm_arch, flags, plat_id, payload_size, csum = struct.unpack(">4s4BHHIQI", data[:28])
    calc_csum = sum(data[:24]) & 0xFFFFFFFF
    is_valid = (magic == MAGIC) and (calc_csum == csum)
    return {
        "valid": is_valid,
        "trt_version": (v_maj, v_min, v_pat, v_bld),
        "sm_arch": sm_arch,
        "hardware_compatible": bool(flags & 1),
        "platform": PLATFORMS.get(plat_id, "unknown"),
        "payload_size": payload_size
    }
