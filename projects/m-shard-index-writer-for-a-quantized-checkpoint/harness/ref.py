import json

CHECKPOINTS = [
    {
        "shard_name": "model-00001-of-00002.safetensors",
        "tensors": {
            "layer.0.weight": {"dtype": "q4_0", "shape": [4096, 4096], "data_offsets": [0, 1048576]},
            "layer.0.scale": {"dtype": "fp16", "shape": [4096, 32], "data_offsets": [1048576, 1114112]}
        }
    },
    {
        "shard_name": "model-00002-of-00002.safetensors",
        "tensors": {
            "layer.1.weight": {"dtype": "q8_0", "shape": [4096, 4096], "data_offsets": [0, 2097152]},
            "layer.1.bias": {"dtype": "fp32", "shape": [4096], "data_offsets": [2097152, 2113536]}
        }
    },
    {
        "shard_name": "model-00001-of-00001.safetensors",
        "tensors": {
            "embed.weight": {"dtype": "q4_k", "shape": [32000, 4096], "data_offsets": [0, 67108864]}
        }
    }
]

def build_index(checkpoint_data):
    weight_map = {}
    metadata = {"total_size": 0, "format": "quantized_v1"}
    for shard in checkpoint_data:
        sname = shard["shard_name"]
        for tname, info in shard["tensors"].items():
            weight_map[tname] = {
                "file": sname,
                "dtype": info["dtype"],
                "shape": info["shape"],
                "offsets": info["data_offsets"]
            }
            metadata["total_size"] = max(metadata["total_size"], info["data_offsets"][1])
    return {"weight_map": weight_map, "metadata": metadata}

def serialize_index(index_dict):
    return json.dumps(index_dict, sort_keys=True, indent=2)

def validate_index(index_dict):
    if "weight_map" not in index_dict or "metadata" not in index_dict:
        return False
    for tname, info in index_dict["weight_map"].items():
        if not all(k in info for k in ("file", "dtype", "shape", "offsets")):
            return False
        if not isinstance(info["offsets"], list) or len(info["offsets"]) != 2:
            return False
        if info["offsets"][0] > info["offsets"][1]:
            return False
    return True
