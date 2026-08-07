import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from overlap import audit

    m = {"buckets_optimized": 0.0}
    tensors = ref.sample_tensors()
    target_size = 3 * 1024 * 1024
    try:
        buckets = audit.optimize_buckets(tensors, target_size)
        if isinstance(buckets, list) and len(buckets) > 0:
            m["buckets_optimized"] = 1.0
    except Exception:
        pass
    return m
