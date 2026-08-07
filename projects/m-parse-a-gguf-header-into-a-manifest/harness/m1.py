import ref


def check(workdir):
    from gguf_parser.header import parse_gguf_header

    out = {
        "headers_matched": 0.0,
        "total_fixtures": float(len(ref.GENERATED_FIXTURES)),
    }
    matched = 0

    for i, fix in enumerate(ref.GENERATED_FIXTURES):
        bin_data = fix["binary"]
        try:
            got = parse_gguf_header(bin_data)
            expected = ref.parse_gguf_header(bin_data)

            if (
                got["version"] == expected["version"]
                and got["tensor_count"] == expected["tensor_count"]
                and got["kv_count"] == expected["kv_count"]
                and got["header_size"] == expected["header_size"]
                and got["metadata"] == expected["metadata"]
                and got["tensors"] == expected["tensors"]
            ):
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"Fixture {i} mismatch: got {got}, expected {expected}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Fixture {i} raised {type(e).__name__}: {str(e)}"

    out["headers_matched"] = float(matched)
    return out
