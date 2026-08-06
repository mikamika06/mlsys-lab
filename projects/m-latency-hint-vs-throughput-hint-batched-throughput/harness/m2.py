import ref


def _match(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if a.keys() != b.keys():
        return False
    for k in a:
        if abs(a[k] - b[k]) > 1e-5:
            return False
    return True


def check(workdir):
    out = {"throughput_match": 0.0, "throughput_ratio": 0.0}
    from cpuhints.scheduler import estimate_throughput

    cores = 16
    bs = ref.BATCH_SIZES

    try:
        lat_res = estimate_throughput(bs, "latency", cores)
        thr_res = estimate_throughput(bs, "throughput", cores)

        ok = True
        if not _match(lat_res, ref.estimate_throughput(bs, "latency", cores)):
            ok = False
        if not _match(thr_res, ref.estimate_throughput(bs, "throughput", cores)):
            ok = False

        out["throughput_match"] = 1.0 if ok else 0.0

        if ok and lat_res.get(16, 0) > 0:
            out["throughput_ratio"] = thr_res[16] / lat_res[16]
    except Exception:
        pass

    return out
