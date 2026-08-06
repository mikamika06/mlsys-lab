import struct

MAGIC = b"TRT\x01"
PLATFORMS = {"linux-x86_64": 1, "linux-aarch64": 2, "windows-x86_64": 3}


def pack_header(trt_ver=(8, 6, 1, 0), sm=80, hw_compat=False, platform="linux-x86_64", payload_size=1024, corrupt_magic=False, corrupt_checksum=False):
    plat_id = PLATFORMS.get(platform, 1)
    magic = b"BAD!" if corrupt_magic else MAGIC
    flags = 1 if hw_compat else 0
    part = struct.pack(">4s4BHHIQ", magic, trt_ver[0], trt_ver[1], trt_ver[2], trt_ver[3], sm, flags, plat_id, payload_size)
    csum = sum(part) & 0xFFFFFFFF
    if corrupt_checksum:
        csum = (csum + 1) & 0xFFFFFFFF
    return part + struct.pack(">I", csum)


def ref_parse_plan_header(data: bytes) -> dict:
    if len(data) < 28:
        return {"valid": False}
    magic, v0, v1, v2, v3, sm, flags, plat_id, payload_size, csum = struct.unpack(">4s4BHHIQI", data[:28])
    plat_map = {1: "linux-x86_64", 2: "linux-aarch64", 3: "windows-x86_64"}
    calc_csum = sum(data[:24]) & 0xFFFFFFFF
    is_valid = (magic == MAGIC) and (calc_csum == csum)
    return {
        "valid": is_valid,
        "trt_version": (v0, v1, v2, v3),
        "sm_arch": sm,
        "hardware_compatible": bool(flags & 1),
        "platform": plat_map.get(plat_id, "unknown"),
        "payload_size": payload_size
    }


def ref_classify_engine(header: dict, runtime_env: dict) -> dict:
    if not header.get("valid", False):
        return {"status": "CORRUPTED_HEADER", "penalty": None}
    if header.get("platform") != runtime_env.get("platform"):
        return {"status": "PLATFORM_MISMATCH", "penalty": None}
    h_ver = header.get("trt_version", (0, 0, 0, 0))
    r_ver = runtime_env.get("trt_version", (0, 0, 0, 0))
    if h_ver[0] != r_ver[0] or h_ver > r_ver:
        return {"status": "VERSION_MISMATCH", "penalty": None}
    h_sm = header.get("sm_arch", 0)
    r_sm = runtime_env.get("sm_arch", 0)
    if h_sm == r_sm:
        return {"status": "OK", "penalty": 1.0}
    if header.get("hardware_compatible", False) and r_sm > h_sm:
        return {"status": "OK", "penalty": round(1.0 + 0.05 * (r_sm - h_sm), 4)}
    return {"status": "INCOMPATIBLE_HARDWARE", "penalty": None}


HEADER_VECTORS = [
    pack_header((8, 6, 1, 0), 80, False, "linux-x86_64", 2048),
    pack_header((8, 5, 3, 0), 75, True, "linux-aarch64", 1024),
    pack_header((9, 0, 0, 0), 90, True, "windows-x86_64", 4096),
    pack_header((8, 6, 0, 0), 86, False, "linux-x86_64", 512),
    pack_header((8, 6, 1, 0), 89, True, "linux-x86_64", 8192),
    pack_header((8, 6, 1, 0), 80, False, "linux-x86_64", 1024, corrupt_magic=True),
    pack_header((8, 6, 1, 0), 80, False, "linux-x86_64", 1024, corrupt_checksum=True),
    b"short_header",
    pack_header((10, 0, 1, 0), 90, False, "linux-x86_64", 16384),
    pack_header((8, 4, 1, 0), 70, True, "windows-x86_64", 2048)
]

CLASSIFY_VECTORS = [
    (pack_header((8, 6, 1, 0), 80, False, "linux-x86_64"), {"trt_version": (8, 6, 1, 0), "sm_arch": 80, "platform": "linux-x86_64"}),
    (pack_header((8, 6, 1, 0), 80, True, "linux-x86_64"), {"trt_version": (8, 6, 1, 0), "sm_arch": 90, "platform": "linux-x86_64"}),
    (pack_header((8, 6, 1, 0), 90, True, "linux-x86_64"), {"trt_version": (8, 6, 1, 0), "sm_arch": 80, "platform": "linux-x86_64"}),
    (pack_header((8, 6, 1, 0), 80, False, "linux-x86_64"), {"trt_version": (8, 6, 1, 0), "sm_arch": 86, "platform": "linux-x86_64"}),
    (pack_header((8, 6, 1, 0), 80, False, "linux-x86_64"), {"trt_version": (9, 0, 0, 0), "sm_arch": 80, "platform": "linux-x86_64"}),
    (pack_header((8, 6, 1, 0), 80, False, "linux-x86_64"), {"trt_version": (8, 5, 0, 0), "sm_arch": 80, "platform": "linux-x86_64"}),
    (pack_header((8, 6, 1, 0), 80, False, "linux-x86_64"), {"trt_version": (8, 6, 1, 0), "sm_arch": 80, "platform": "linux-aarch64"}),
    (pack_header((8, 6, 1, 0), 80, False, "linux-x86_64", corrupt_magic=True), {"trt_version": (8, 6, 1, 0), "sm_arch": 80, "platform": "linux-x86_64"}),
    (pack_header((8, 6, 1, 0), 80, False, "linux-x86_64", corrupt_checksum=True), {"trt_version": (8, 6, 1, 0), "sm_arch": 80, "platform": "linux-x86_64"}),
    (b"truncated", {"trt_version": (8, 6, 1, 0), "sm_arch": 80, "platform": "linux-x86_64"}),
    (pack_header((8, 6, 1, 0), 80, True, "linux-x86_64"), {"trt_version": (8, 6, 1, 0), "sm_arch": 80, "platform": "linux-x86_64"}),
    (pack_header((8, 6, 0, 0), 80, False, "linux-x86_64"), {"trt_version": (8, 6, 2, 0), "sm_arch": 80, "platform": "linux-x86_64"})
]
