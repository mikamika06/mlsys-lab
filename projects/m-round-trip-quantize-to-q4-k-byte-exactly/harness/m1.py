import ref

def check(workdir):
    from q4k.quant import round_trip_q4_k
    out = {"byte_exact_fraction": 0.0}
    ok = 0
    total = len(ref.TEST_BLOCKS)
    for i, block in enumerate(ref.TEST_BLOCKS):
        got = round_trip_q4_k(block)
        if got == 1.0:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"block {i}: expected 1.0 byte-exact round trip, got {got}"
    out["byte_exact_fraction"] = float(ok / total)
    return out
