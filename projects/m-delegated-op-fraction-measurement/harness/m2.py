import ref

def check(workdir):
    try:
        from delegate_measure.partitioner import partition_xnnpack, partition_coreml
    except ImportError:
        return {"match_xnnpack": 0.0, "match_coreml": 0.0}

    x_ok = 0
    c_ok = 0

    for ops in ref.MODELS:
        try:
            if partition_xnnpack(ops) == ref.partition_xnnpack(ops):
                x_ok += 1
            if partition_coreml(ops) == ref.partition_coreml(ops):
                c_ok += 1
        except Exception:
            pass

    return {
        "match_xnnpack": 1.0 if x_ok == len(ref.MODELS) else 0.0,
        "match_coreml": 1.0 if c_ok == len(ref.MODELS) else 0.0
    }
