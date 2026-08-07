import numpy as np
from gguf_shard.model import Model

def split(model, max_bytes):
    shards = []
    cur_tensors = {}
    cur_bytes = 0
    for k in sorted(model.tensors.keys()):
        t = model.tensors[k]
        sz = t.nbytes
        if cur_bytes + sz > max_bytes and cur_tensors:
            shards.append(cur_tensors)
            cur_tensors = {}
            cur_bytes = 0
        cur_tensors[k] = t
        cur_bytes += sz
    if cur_tensors:
        shards.append(cur_tensors)

    res = []
    for i, t_dict in enumerate(shards):
        meta = model.metadata.copy()
        meta["split.no"] = i
        meta["split.count"] = len(shards)
        meta["split.checksum"] = sum(float(np.sum(v)) for v in t_dict.values())
        res.append(Model(meta, t_dict))
    return res

def verify_shard(shard):
    expected = shard.metadata.get("split.checksum")
    if expected is None:
        return False
    actual = sum(float(np.sum(v)) for v in shard.tensors.values())
    return abs(expected - actual) < 1e-5

def reassemble(shards):
    shards_sorted = sorted(shards, key=lambda s: s.metadata.get("split.no", 0))
    count = shards_sorted[0].metadata.get("split.count", len(shards))
    if len(shards) != count:
        raise ValueError("Missing shards")
    for i, s in enumerate(shards_sorted):
        if s.metadata.get("split.no", -1) != i:
            raise ValueError("Gap in shards")
        if not verify_shard(s):
            raise ValueError("Corrupt shard")

    meta = shards_sorted[0].metadata.copy()
    meta.pop("split.no", None)
    meta.pop("split.count", None)
    meta.pop("split.checksum", None)

    tensors = {}
    for s in shards_sorted:
        tensors.update(s.tensors)
    return Model(meta, tensors)
