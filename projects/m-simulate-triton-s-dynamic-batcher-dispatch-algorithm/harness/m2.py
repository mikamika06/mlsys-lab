import ref
import sys

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from triton_batcher.metrics import measure_metrics
    except ImportError:
        return {"_note": "could not import measure_metrics"}

    out = {"matches": 0.0, "total": 2.0}
    ok = 0
    configs = [([4], 50000), ([2, 4], 2000)]

    for pref, delay in configs:
        disps = ref.simulate(ref.ARRIVALS[:100], 8, pref, delay, ref.dummy_compute_fn)
        want = ref.measure_metrics(ref.ARRIVALS[:100], disps, ref.dummy_compute_fn)
        try:
            got = measure_metrics(ref.ARRIVALS[:100], disps, ref.dummy_compute_fn)

            t_diff = abs(want["throughput_req_sec"] - got.get("throughput_req_sec", -1))
            p_diff = abs(want["p99_queue_delay_us"] - got.get("p99_queue_delay_us", -1))

            if t_diff < 1e-4 and p_diff < 1e-4:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"failed for delay={delay}. want {want}, got {got}"
        except Exception as e:
            out["_note"] = f"crash on delay={delay}: {e}"
            break

    sys.path.pop(0)
    out["matches"] = float(ok)
    return out
