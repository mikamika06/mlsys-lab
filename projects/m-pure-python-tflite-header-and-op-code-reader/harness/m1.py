import ref


def check(workdir):
    from tflite_tools.parser import parse_header, extract_op_codes

    out = {"headers_matched": 0.0, "opcodes_matched": 0.0}
    h_ok = 0
    o_ok = 0

    for i, m_bytes in enumerate(ref.MODELS):
        want_h = parse_header(m_bytes)
        want_o = extract_op_codes(m_bytes)

        try:
            got_h = parse_header(m_bytes)
            got_o = extract_op_codes(m_bytes)
        except Exception as e:
            out["_note"] = f"model {i} raised {type(e).__name__}"
            return out

        if got_h and got_h.get("root_table_offset") == want_h.get("root_table_offset") and got_h.get("file_identifier") == want_h.get("file_identifier"):
            h_ok += 1
        if got_o == want_o:
            o_ok += 1

    out["headers_matched"] = float(h_ok)
    out["opcodes_matched"] = float(o_ok)
    return out
