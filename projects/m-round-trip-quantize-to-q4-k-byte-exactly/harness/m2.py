import ref

def check(workdir):
    from q4k.mse import locate_dominating_subblock
    out = {"subblock_match": 0.0}
    ok = 0
    total = len(ref.TEST_BLOCKS)
    for i, block in enumerate(ref.TEST_BLOCKS):
        want = ref.locate_dominating_subblock(block)
        got = locate_dominating_subblock(block)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"block {i}: got subblock {got}, reference {want}"
    out["subblock_match"] = float(1.0 if ok == total else 0.0)
    return out
