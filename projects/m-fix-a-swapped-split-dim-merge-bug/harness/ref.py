import numpy as np

CONFIGS = [
    {
        "param_name": "model.layers.0.self_attn.qkv_proj.weight",
        "shape": (3072, 768),
        "shards": 2,
        "split_dim": 0
    },
    {
        "param_name": "model.layers.0.mlp.dense_h_to_4h.weight",
        "shape": (3072, 768),
        "shards": 4,
        "split_dim": 0
    }
]

def reconstruct_mapping(config):
    name = config["param_name"]
    shape = config["shape"]
    shards = config["shards"]
    split_dim = config["split_dim"]
    shard_shape = list(shape)
    shard_shape[split_dim] = shape[split_dim] // shards
    res = []
    for i in range(shards):
        res.append({
            "shard_id": i,
            "name": f"{name}.shard_{i}",
            "shape": tuple(shard_shape),
            "split_dim": split_dim
        })
    return res

def convert_tensor(shards_data, split_dim):
    return np.concatenate(shards_data, axis=split_dim)

def fix_merge(tensor, split_dim):
    return tensor
