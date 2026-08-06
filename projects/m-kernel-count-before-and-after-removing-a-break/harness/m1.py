import ref

def check(workdir):
    from fusion.parser import parse_kernels, count_kernels

    out = {"parse_match": 0.0, "count_match": 0.0}
    try:
        got_p1 = parse_kernels(ref.CODE_WITH_BREAK)
        got_p2 = parse_kernels(ref.CODE_FUSED)
        want_p1 = ref.parse_kernels(ref.CODE_WITH_BREAK)
        want_p2 = ref.parse_kernels(ref.CODE_FUSED)

        if got_p1 == want_p1 and got_p2 == want_p2:
            out["parse_match"] = 1.0

        got_c1 = count_kernels(ref.CODE_WITH_BREAK)
        got_c2 = count_kernels(ref.CODE_FUSED)
        want_c1 = ref.count_kernels(ref.CODE_WITH_BREAK)
        want_c2 = ref.count_kernels(ref.CODE_FUSED)

        if got_c1 == want_c1 and got_c2 == want_c2:
            out["count_match"] = 1.0

    except Exception as e:
        out["_note"] = f"Failed to parse kernels: {e}"

    return out
