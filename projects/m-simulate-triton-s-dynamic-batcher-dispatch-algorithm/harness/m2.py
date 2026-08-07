import ref
import sys

def check(workdir):
    out = {"matches_reference": 0.0, "total": 2.0}
    sys.path.insert(0, workdir)
    try:
        from triton.metrics import calculate_metrics
        ok = 0
        for delay in [1000, 50000]:
            batches = ref.simulate(ref.ARRIVALS[:100], 8, [4], delay, ref.compute_fn)
            want = ref.calculate_metrics(ref.ARRIVALS[:100], batches, ref.compute_fn)
            got = calculate_metrics(ref.ARRIVALS[:100], batches, ref.compute_fn)

            t_diff = abs(want["throughput"] - got["throughput"])
            p_diff = abs(want["p99_queue_delay"] - got["p99_queue_delay"])

            if t_diff < 1e-4 and p_diff < 1e-4:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"mismatch at delay {delay}: got {got}, want {want}"
        out["matches_reference"] = float(ok)
    except Exception as e:
        out["_note"] = f"error: {e}"
    finally:
        sys.path.pop(0)
    return out
