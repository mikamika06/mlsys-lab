import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from kvquant.parser import parse_gguf_kv_params

    out = {"headers_matched": 0.0}
    matched = 0

    for i, data in enumerate(ref.BINARY_FIXTURES):
        expected = ref.GGUF_TEST_CASES[i]
        try:
            got = parse_gguf_kv_params(data)
            if got == expected:
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"Header {i} mismatch. Expected {expected}, got {got}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Header {i} failed with exception: {e}"

    out["headers_matched"] = float(matched)
    return out
