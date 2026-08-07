import ref
import sys

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from triton_batcher.simulate import simulate
    except ImportError:
        return {"_note": "could not import simulate"}

    out = {"matches": 0.0, "total": 2.0}
    ok = 0
    configs = [([4], 50000), ([2, 4], 2000)]

    for pref, delay in configs:
        want = ref.simulate(ref.ARRIVALS[:100], 8, pref, delay, ref.dummy_compute_fn)
        try:
            got = simulate(ref.ARRIVALS[:100], 8, pref, delay, ref.dummy_compute_fn)
            if want == got:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"failed for pref={pref} delay={delay}. want {want[:2]}... got {got[:2] if got else []}..."
        except Exception as e:
            out["_note"] = f"crash on pref={pref} delay={delay}: {e}"
            break

    sys.path.pop(0)
    out["matches"] = float(ok)
    return out
