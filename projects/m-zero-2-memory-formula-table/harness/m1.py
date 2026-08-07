import ref

def check(workdir):
    from zeromem.formula import compute_memory_table
    out = {"formulas_matched": 0.0}
    test_cases = [
        (1024 * 1024 * 100, 4, 4),
        (1024 * 1024 * 512, 8, 4),
        (1024 * 1024 * 256, 2, 4)
    ]
    matched = True
    for pb, ws, op in test_cases:
        want = ref.compute_memory_table(pb, ws, op)
        try:
            got = compute_memory_table(pb, ws, op)
        except Exception as e:
            matched = False
            out["_note"] = f"raised exception: {e}"
            break
        if got != want:
            matched = False
            out["_note"] = f"got {got}, want {want}"
            break
    if matched:
        out["formulas_matched"] = 1.0
    return out
