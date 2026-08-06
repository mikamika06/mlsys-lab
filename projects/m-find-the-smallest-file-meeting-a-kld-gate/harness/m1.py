import ref


def check(workdir):
    from quant.gate import find_smallest_file

    out = {"candidates_matched": 0.0}
    ok = 0
    for i, candidates in enumerate(ref.CANDIDATES_SET):
        max_kld = ref.MAX_KLDS[i]
        valid = [c for c in candidates if c["kld"] <= max_kld]
        ref_best = min(valid, key=lambda x: x["size"]) if valid else None

        got = find_smallest_file(candidates, max_kld)
        if got == ref_best:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {ref_best}"

    out["candidates_matched"] = float(ok)
    return out
