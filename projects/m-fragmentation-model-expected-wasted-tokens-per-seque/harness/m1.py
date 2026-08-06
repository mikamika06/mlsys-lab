import ref


def check(workdir):
    from frag.model import expected_wasted_tokens, optimal_block_size

    out = {"wasted_match": 0.0, "optimal_match": 0.0}
    wasted_ok = True
    for hist in ref.HISTOGRAMS:
        for bs in ref.CANDIDATES:
            want = ref.expected_wasted_tokens(hist, bs)
            got = expected_wasted_tokens(hist, bs)
            if abs(float(got) - float(want)) > 1e-5:
                wasted_ok = False
                out["_note"] = f"wasted_tokens mismatch for hist {hist}, bs {bs}: got {got}, want {want}"
                break
        if not wasted_ok:
            break

    if wasted_ok:
        out["wasted_match"] = 1.0

    optimal_ok = True
    for hist in ref.HISTOGRAMS:
        want = ref.optimal_block_size(hist, ref.CANDIDATES, 2.0)
        got = optimal_block_size(hist, ref.CANDIDATES, 2.0)
        if got != want:
            optimal_ok = False
            out["_note"] = f"optimal_block_size mismatch for hist {hist}: got {got}, want {want}"
            break

    if optimal_ok:
        out["optimal_match"] = 1.0

    return out
