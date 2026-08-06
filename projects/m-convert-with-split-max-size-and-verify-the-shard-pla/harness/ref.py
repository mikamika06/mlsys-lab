import numpy as np

MODELS = [
    {
        "vocab": {"tokens": ["<pad>", "<s>", "</s>", "foo", "bar", "baz"], "scores": [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]},
        "tensors": [
            {"name": "token_embd.weight", "shape": (6, 16), "dtype": "float32"},
            {"name": "blk.0.attn_q.weight", "shape": (16, 16), "dtype": "float32"},
            {"name": "blk.0.attn_k.weight", "shape": (16, 16), "dtype": "float32"},
            {"name": "output.weight", "shape": (6, 16), "dtype": "float32"},
        ],
        "max_bytes": 1024,
    },
    {
        "vocab": {"tokens": [f"tok_{i}" for i in range(100)], "scores": [float(i) for i in range(100)]},
        "tensors": [
            {"name": "embed.weight", "shape": (100, 64), "dtype": "float32"},
            {"name": "layer.0.weight", "shape": (64, 64), "dtype": "float32"},
            {"name": "layer.1.weight", "shape": (64, 64), "dtype": "float32"},
            {"name": "layer.2.weight", "shape": (64, 64), "dtype": "float32"},
        ],
        "max_bytes": 12000,
    },
    {
        "vocab": {"tokens": ["a", "b", "c"], "scores": [1.0, 1.0, 1.0]},
        "tensors": [
            {"name": "t1", "shape": (100, 100), "dtype": "float32"},
            {"name": "t2", "shape": (200, 200), "dtype": "float32"},
        ],
        "max_bytes": 50000,
    },
    {
        "vocab": {"tokens": ["x", "y"], "scores": [0.5, 0.5]},
        "tensors": [
            {"name": "a", "shape": (10,), "dtype": "float32"},
            {"name": "b", "shape": (20,), "dtype": "float32"},
            {"name": "c", "shape": (30,), "dtype": "float32"},
        ],
        "max_bytes": 200,
    },
]

DTYPE_SIZES = {"float32": 4, "float16": 2, "int32": 4}

def tensor_size(t):
    size = DTYPE_SIZES[t["dtype"]]
    for dim in t["shape"]:
        size *= dim
    return size

def plan_shards(model_cfg):
    max_bytes = model_cfg["max_bytes"]
    shards = []
    curr_shard = []
    curr_size = 0
    for t in model_cfg["tensors"]:
        sz = tensor_size(t)
        if curr_shard and (curr_size + sz > max_bytes):
            shards.append({"tensors": curr_shard, "size": curr_size})
            curr_shard = []
            curr_size = 0
        curr_shard.append(t["name"])
        curr_size += sz
    if curr_shard:
        shards.append({"tensors": curr_shard, "size": curr_size})
    return shards

def convert_vocab_only(vocab_data):
    tokens = vocab_data["tokens"]
    scores = vocab_data["scores"]
    encoded = []
    for t, s in zip(tokens, scores):
        encoded.append({"token": t.encode("utf-8").hex(), "score": float(s)})
    return {"vocab_size": len(tokens), "entries": encoded}
