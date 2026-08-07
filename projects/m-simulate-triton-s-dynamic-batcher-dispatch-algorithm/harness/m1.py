import ref
import sys

def check(workdir):
    out = {"matches_reference": 0.0, "total": 2.0}
    sys.path.insert(0, workdir)
    try:
        from triton.simulate import simulate
        ok = 0
        for delay in [1000, 50000]:
            want = ref.simulate(ref.ARRIVALS[:100], 8, [4], delay, ref.compute_fn)
            got = simulate(ref.ARRIVALS[:100], 8, [4], delay, ref.compute_fn)
            if want == got:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"mismatch at delay {delay}. expected {want[:2]}... got {got[:2]}..."
        out["matches_reference"] = float(ok)
    except Exception as e:
        out["_note"] = f"error: {e}"
    finally:
        sys.path.pop(0)
    return out
