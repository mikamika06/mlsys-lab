from mlsys import scorers


def _oracle(data):
    view = memoryview(data)
    ints = view.cast("i")
    bytes_view = ints.cast("B")
    floats = bytes_view.cast("f")
    return floats.cast("B").tobytes()


def grade(sol, fx) -> dict:
    cases = [
        bytes.fromhex("0000803f00000040"),
        bytes.fromhex("000000000000803f0000004000004040"),
        bytes(range(32)),
        bytes((i * 37) % 256 for i in range(48)),
    ]

    ref_parts = []
    got_parts = []

    for case in cases:
        try:
            got = sol.reinterpret_roundtrip(case)
        except Exception:
            return {"byte_exact_fraction": 0.0}
        if not isinstance(got, (bytes, bytearray)):
            return {"byte_exact_fraction": 0.0}
        ref_parts.append(_oracle(case))
        got_parts.append(bytes(got))

    return {
        "byte_exact_fraction": scorers.byte_exact_fraction(
            b"".join(ref_parts),
            b"".join(got_parts),
        )
    }
