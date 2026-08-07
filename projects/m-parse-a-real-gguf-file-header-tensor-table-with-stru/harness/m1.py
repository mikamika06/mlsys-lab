import ref


def check(workdir):
    from ggufparser.parser import parse_gguf_header
    data, expected_tensors = ref.generate_mock_gguf()

    try:
        header, tensors = parse_gguf_header(data)
    except Exception as e:
        return {"header_matched": 0.0, "tensors_matched": 0.0, "_note": f"Exception: {e}"}

    h_ok = 1.0 if header["version"] == 3 and header["tensor_count"] == 2 else 0.0

    t_ok = 1.0
    if len(tensors) != len(expected_tensors):
        t_ok = 0.0
    else:
        for got, want in zip(tensors, expected_tensors):
            if got["name"] != want["name"] or got["type"] != want["type"] or got["dims"] != want["dims"]:
                t_ok = 0.0
                break

    return {"header_matched": h_ok, "tensors_matched": t_ok}
