import struct


def parse_header(data: bytes) -> dict:
    if len(data) < 24:
        raise ValueError("Buffer too short for header")
    magic, version, sm_maj, sm_min, tactics, opt = struct.unpack("<4sIIIII", data[:24])
    if magic != b"TRTC":
        raise ValueError("Invalid magic bytes")

    return {
        "version": version,
        "sm_major": sm_maj,
        "sm_minor": sm_min,
        "tactic_sources": tactics,
        "opt_level": opt
    }


def is_reusable(cache_header: dict, builder_config: dict) -> bool:
    if cache_header["version"] != builder_config["version"]:
        return False
    if cache_header["sm_major"] != builder_config["sm_major"]:
        return False
    if cache_header["sm_minor"] != builder_config["sm_minor"]:
        return False
    if cache_header["opt_level"] != builder_config["opt_level"]:
        return False

    cache_tactics = cache_header["tactic_sources"]
    builder_tactics = builder_config["tactic_sources"]
    if (cache_tactics & ~builder_tactics) != 0:
        return False

    return True
