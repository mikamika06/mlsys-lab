import ref

def check(workdir):
    from optimizer.chunking import calculate_prefix_savings

    trace = ref.generate_trace()
    sizes = [4, 8, 16, 32]
    out = {"savings_matched": 0.0}
    ok = 0

    for s in sizes:
        want = ref.calculate_prefix_savings(trace, s)
        try:
            got = calculate_prefix_savings(trace, s)
            if got == want:
                ok += 1
            else:
                out["_note"] = f"size {s}: got {got}, want {want}"
        except Exception as e:
            out["_note"] = f"size {s} raised {type(e).__name__}"

    out["savings_matched"] = float(ok)
    return out
