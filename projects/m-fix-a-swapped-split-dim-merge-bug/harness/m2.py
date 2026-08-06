import ref
import numpy as np

def check(workdir):
    from ucp.converter import convert_tensor
    out = {"outputs_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        shape = cfg["shape"]
        split_dim = cfg["split_dim"]
        shards = cfg["shards"]
        rng = np.random.default_rng(42 + i)
        full_tensor = rng.standard_normal(shape)
        shard_shape = list(shape)
        shard_shape[split_dim] = shape[split_dim] // shards
        shards_data = np.split(full_tensor, shards, axis=split_dim)
        want = ref.convert_tensor(shards_data, split_dim)
        got = convert_tensor(list(shards_data), split_dim)
        if np.allclose(got, want):
            ok += 1
        else:
            out["_note"] = f"tensor conversion mismatch at index {i}"
    if ok == len(ref.CONFIGS):
        out["outputs_matched"] = 1.0
    return out
