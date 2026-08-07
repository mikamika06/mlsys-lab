import os
import json
import torch

def parse_structure(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    metadata = {}
    shards = []
    for f in sorted(files):
        if f.endswith(".json"):
            with open(os.path.join(checkpoint_dir, f), "r") as mf:
                metadata = json.load(mf)
        elif f.endswith(".pt") or f.endswith(".bin"):
            shards.append(f)
    return {"metadata": metadata, "shards": shards}

def map_sharding(state_dict, world_size):
    mapped = {}
    for k, v in state_dict.items():
        mapped[k] = {"shape": list(v.shape), "world_size": world_size}
    return mapped

def convert_to_portable(checkpoint_dir, output_path):
    struct = parse_structure(checkpoint_dir)
    combined = {}
    for shard in struct["shards"]:
        path = os.path.join(checkpoint_dir, shard)
        data = torch.load(path, map_location="cpu")
        for k, v in data.items():
            if k not in combined:
                combined[k] = []
            combined[k].append(v)

    final_state = {}
    for k, chunks in combined.items():
        final_state[k] = torch.cat(chunks, dim=0)

    torch.save(final_state, output_path)
    return output_path

def restore_from_portable(portable_path, target_world_size, rank):
    state = torch.load(portable_path, map_location="cpu")
    restored = {}
    for k, v in state.items():
        dim_size = v.shape[0]
        chunk_size = dim_size // target_world_size
        start = rank * chunk_size
        end = start + chunk_size
        restored[k] = v[start:end].clone()
    return restored
