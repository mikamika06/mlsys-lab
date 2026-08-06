import numpy as np


def merge_imatrices(shards):
    """Merge multiple imatrix shards using sample counts as weights."""
    if not shards:
        return {"count": 0, "data": {}}
    total = sum(s["count"] for s in shards)
    if total == 0:
        return {"count": 0, "data": {}}
    keys = shards[0]["data"].keys()
    merged = {}
    for k in keys:
        acc = np.zeros_like(shards[0]["data"][k], dtype=np.float64)
        for s in shards:
            acc += np.asarray(s["data"][k], dtype=np.float64) * s["count"]
        merged[k] = (acc / total).astype(np.float32)
    return {"count": total, "data": merged}
