import sys


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"divergence_match": 0.0, "threshold_match": 0.0}
    try:
        from mps_mem.mock_device import MPSDevice
        from mps_mem.divergence import log_divergence
        from mps_mem.threshold import find_oom_threshold
        import ref

        ok = 0
        for ops in ref.WORKLOADS:
            want = ref.expected_divergence(MPSDevice, ops)
            got = log_divergence(MPSDevice(), ops)
            if want == got:
                ok += 1
        if ok == len(ref.WORKLOADS):
            out["divergence_match"] = 1.0

        dev = MPSDevice(total_mem=54321)
        best, rec = find_oom_threshold(dev)
        if best == 54321 and rec == int(54321 * 0.7):
            out["threshold_match"] = 1.0
    except Exception as e:
        out["_note"] = f"Error: {e}"
    finally:
        sys.path.pop(0)
    return out
