import ref


def check(workdir):
    from tlog.unsupported import extract_unsupported_op
    out = {"op_matched": 0.0}
    ok = True
    for log_text, expected_op in ref.LOG_SNIPPETS:
        got = extract_unsupported_op(log_text)
        if got != expected_op:
            ok = False
            break
    if ok:
        out["op_matched"] = 1.0
    return out
