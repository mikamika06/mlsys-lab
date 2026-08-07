import json


def build_shard_index(checkpoint_data):
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
