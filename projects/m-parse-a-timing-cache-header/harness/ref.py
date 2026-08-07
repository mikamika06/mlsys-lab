import struct


def make_header(version, sm_maj, sm_min, tactics, opt, magic=b"TRTC"):
    return struct.pack("<4sIIIII", magic, version, sm_maj, sm_min, tactics, opt)


HEADERS = [
    make_header(8600, 8, 9, 7, 3),
    make_header(8500, 8, 0, 3, 3),
    make_header(8600, 8, 9, 7, 3, b"BAD!"),
    b"short_file_data_here_",
    make_header(10000, 9, 0, 1, 1),
]


def ref_parse(data: bytes) -> dict:
    if len(data) < 24:
        raise ValueError
    magic, v, smaj, smin, t, o = struct.unpack("<4sIIIII", data[:24])
    if magic != b"TRTC":
        raise ValueError
    return {
        "version": v,
        "sm_major": smaj,
        "sm_minor": smin,
        "tactic_sources": t,
        "opt_level": o
    }


CONFIGS = [
    (
        {"version": 8600, "sm_major": 8, "sm_minor": 9, "tactic_sources": 3, "opt_level": 3},
        {"version": 8600, "sm_major": 8, "sm_minor": 9, "tactic_sources": 3, "opt_level": 3}
    ),
    (
        {"version": 8600, "sm_major": 8, "sm_minor": 9, "tactic_sources": 1, "opt_level": 3},
        {"version": 8600, "sm_major": 8, "sm_minor": 9, "tactic_sources": 3, "opt_level": 3}
    ),
    (
        {"version": 8600, "sm_major": 8, "sm_minor": 9, "tactic_sources": 7, "opt_level": 3},
        {"version": 8600, "sm_major": 8, "sm_minor": 9, "tactic_sources": 3, "opt_level": 3}
    ),
    (
        {"version": 8500, "sm_major": 8, "sm_minor": 9, "tactic_sources": 3, "opt_level": 3},
        {"version": 8600, "sm_major": 8, "sm_minor": 9, "tactic_sources": 3, "opt_level": 3}
    ),
    (
        {"version": 8600, "sm_major": 8, "sm_minor": 9, "tactic_sources": 3, "opt_level": 2},
        {"version": 8600, "sm_major": 8, "sm_minor": 9, "tactic_sources": 3, "opt_level": 3}
    )
]


def ref_reusable(ch, bc):
    if ch["version"] != bc["version"]:
        return False
    if ch["sm_major"] != bc["sm_major"]:
        return False
    if ch["sm_minor"] != bc["sm_minor"]:
        return False
    if ch["opt_level"] != bc["opt_level"]:
        return False
    return (ch["tactic_sources"] & ~bc["tactic_sources"]) == 0
