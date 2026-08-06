import ref

def check(workdir):
    from capacity.disk import measure_read_amplification
    out = {"amplification_matched": 0.0, "total": float(len(ref.DISK_WORKLOADS))}
    ok = 0
    for i, w in enumerate(ref.DISK_WORKLOADS):
        want = ref.build_disk_measure(w["block_size"], w["requests"])
        got = measure_read_amplification(w["block_size"], w["requests"])
        if got is not None and abs(got["read_amplification"] - want["read_amplification"]) < 1e-4 and got["physical_bytes"] == want["physical_bytes"]:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"workload {i}: got {got}, want {want}"
    out["amplification_matched"] = float(ok)
    return out
