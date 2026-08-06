import numpy as np


MODEL_CONFIG = {
    "layers": {
        "blk.0.attn_q.weight": 64,
        "blk.0.attn_k.weight": 64,
        "blk.0.attn_v.weight": 64,
        "blk.0.ffn_down.weight": 128,
    }
}


def get_dataset():
    rng = np.random.default_rng(2026)
    batches = []
    counts = [120, 350, 230]
    for c in counts:
        acts = {}
        for layer, dim in MODEL_CONFIG["layers"].items():
            raw = rng.normal(loc=0.1, scale=1.5, size=(c, dim)).astype(np.float32)
            acts[layer] = raw
        batches.append({"count": c, "activations": acts})
    return MODEL_CONFIG, batches


def reference_compute(activations):
    res = {}
    for k, act in activations.items():
        arr = np.asarray(act, dtype=np.float32)
        res[k] = np.mean(np.square(arr), axis=0)
    return res


def reference_merge(shards):
    if not shards:
        return {"count": 0, "data": {}}
    total = sum(s["count"] for s in shards)
    keys = shards[0]["data"].keys()
    merged = {}
    for k in keys:
        acc = np.zeros_like(shards[0]["data"][k], dtype=np.float64)
        for s in shards:
            acc += np.asarray(s["data"][k], dtype=np.float64) * s["count"]
        merged[k] = (acc / total).astype(np.float32)
    return {"count": total, "data": merged}
