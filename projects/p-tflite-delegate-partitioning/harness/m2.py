import os
import ref

def check(workdir):
    m = {"unsupported_found": 0.0}
    path = ref.create_dummy_model(workdir)
    try:
        from edge.partitioner import find_unsupported_ops
        ops = find_unsupported_ops(path)
        if isinstance(ops, list) and len(ops) > 0:
            m["unsupported_found"] = 1.0
    except Exception:
        pass
    return m
