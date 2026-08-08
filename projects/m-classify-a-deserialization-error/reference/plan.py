import struct


def parse_header(engine_bytes: bytes) -> dict:
    """
    Parse the first 20 bytes of a .plan file.
    Format is 5 32-bit little-endian fields:
    - magic (4 bytes)
    - trt_version (uint32)
    - build_sm (uint32)
    - hw_compat (uint32)
    - os_id (uint32)

    Returns a dict with keys: 'magic', 'trt_version', 'build_sm', 'hw_compat', 'os_id'.
    """
    magic, trt, sm, compat, os_id = struct.unpack("<4sIIII", engine_bytes[:20])
    return {
        "magic": magic,
        "trt_version": trt,
        "build_sm": sm,
        "hw_compat": compat,
        "os_id": os_id
    }


def diagnose_load(engine_bytes: bytes, env_trt: int, env_sm: int, env_os: int) -> dict:
    """
    Diagnoses engine load compatibility and performance penalty.
    """
    if len(engine_bytes) < 20:
        return {"status": "ERR_TRUNCATED", "penalty": 0.0}

    h = parse_header(engine_bytes)

    if h["magic"] != b'TRT\x00':
        return {"status": "ERR_MAGIC", "penalty": 0.0}

    if h["trt_version"] != env_trt:
        return {"status": "ERR_TRT_VERSION", "penalty": 0.0}

    if h["os_id"] != env_os:
        return {"status": "ERR_OS", "penalty": 0.0}

    if h["build_sm"] != env_sm:
        if h["hw_compat"] == 0:
            return {"status": "ERR_SM_ARCH", "penalty": 0.0}
        else:
            if h["build_sm"] < 80 or env_sm < 80:
                return {"status": "ERR_SM_ARCH_UNSUPPORTED", "penalty": 0.0}
            return {"status": "OK", "penalty": 8.5}
    else:
        if h["hw_compat"] == 1:
            return {"status": "OK", "penalty": 3.0}
        return {"status": "OK", "penalty": 0.0}
