import ref


def check(workdir):
    from trtcache.parser import parse_header

    out = {"headers_matched": 0.0, "headers": float(len(ref.HEADERS))}
    ok = 0
    for i, data in enumerate(ref.HEADERS):
        want_err = False
        try:
            want = ref.ref_parse(data)
        except ValueError:
            want_err = True

        got_err = False
        try:
            got = parse_header(data)
        except ValueError:
            got_err = True
            got = None

        if want_err and got_err:
            ok += 1
        elif not want_err and not got_err and want == got:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"header {i}: want_err={want_err}, got_err={got_err}, want={want if not want_err else 'N/A'}"

    out["headers_matched"] = float(ok)
    return out
