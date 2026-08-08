import ref


def check(workdir):
    from exportcheck.sigdef import parse_signature_def

    out = {"sigdefs_matched": 0.0}
    ok = 0
    for i, item in enumerate(ref.SIGNATURE_TESTS):
        binary_data = ref.make_binary_sigdef(item["inputs"], item["outputs"])
        want = item
        try:
            got = parse_signature_def(binary_data)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"test {i}: got {got}, want {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"test {i} raised {type(e).__name__}: {str(e)[:100]}"
    out["sigdefs_matched"] = float(ok)
    return out
