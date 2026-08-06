import ref

def check(workdir):
    from dequant import dequantize_q4_k

    out = {"matches": 0.0}
    ok = 0
    for i, block in enumerate(ref.Q4_FIXTURES):
        want = ref.dequantize_q4_k(block)
        got = dequantize_q4_k(block)
        if len(want) != len(got):
            out["_note"] = f"fixture {i}: length mismatch"
            break
        diff = max(abs(w - g) for w, g in zip(want, got))
        if diff < 1e-4:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"fixture {i}: max diff {diff}"

    out["matches"] = float(ok)
    return out
