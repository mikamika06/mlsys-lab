import struct


def generate_fixtures():
    fixtures = []

    # 1. Truncated
    fixtures.append({
        "engine": b"TRT\x00\x08",
        "env": (8, 80, 1),
        "want_status": "ERR_TRUNCATED",
        "want_penalty": 0.0
    })

    # 2. Bad magic
    fixtures.append({
        "engine": struct.pack("<4sIIII", b"BAD\x00", 8, 80, 0, 1),
        "env": (8, 80, 1),
        "want_status": "ERR_MAGIC",
        "want_penalty": 0.0
    })

    # 3. TRT version mismatch
    fixtures.append({
        "engine": struct.pack("<4sIIII", b"TRT\x00", 7, 80, 0, 1),
        "env": (8, 80, 1),
        "want_status": "ERR_TRT_VERSION",
        "want_penalty": 0.0
    })

    # 4. OS mismatch
    fixtures.append({
        "engine": struct.pack("<4sIIII", b"TRT\x00", 8, 80, 0, 2),
        "env": (8, 80, 1),
        "want_status": "ERR_OS",
        "want_penalty": 0.0
    })

    # 5. SM mismatch, hw_compat=0
    fixtures.append({
        "engine": struct.pack("<4sIIII", b"TRT\x00", 8, 80, 0, 1),
        "env": (8, 89, 1),
        "want_status": "ERR_SM_ARCH",
        "want_penalty": 0.0
    })

    # 6. SM mismatch, hw_compat=1, build_sm < 80
    fixtures.append({
        "engine": struct.pack("<4sIIII", b"TRT\x00", 8, 75, 1, 1),
        "env": (8, 89, 1),
        "want_status": "ERR_SM_ARCH_UNSUPPORTED",
        "want_penalty": 0.0
    })

    # 7. SM mismatch, hw_compat=1, env_sm < 80
    fixtures.append({
        "engine": struct.pack("<4sIIII", b"TRT\x00", 8, 89, 1, 1),
        "env": (8, 75, 1),
        "want_status": "ERR_SM_ARCH_UNSUPPORTED",
        "want_penalty": 0.0
    })

    # 8. SM mismatch, hw_compat=1, both >= 80
    fixtures.append({
        "engine": struct.pack("<4sIIII", b"TRT\x00", 8, 80, 1, 1),
        "env": (8, 89, 1),
        "want_status": "OK",
        "want_penalty": 8.5
    })

    # 9. Exact match, hw_compat=0
    fixtures.append({
        "engine": struct.pack("<4sIIII", b"TRT\x00", 8, 80, 0, 1),
        "env": (8, 80, 1),
        "want_status": "OK",
        "want_penalty": 0.0
    })

    # 10. Exact match, hw_compat=1
    fixtures.append({
        "engine": struct.pack("<4sIIII", b"TRT\x00", 8, 80, 1, 1),
        "env": (8, 80, 1),
        "want_status": "OK",
        "want_penalty": 3.0
    })

    return fixtures


FIXTURES = generate_fixtures()
