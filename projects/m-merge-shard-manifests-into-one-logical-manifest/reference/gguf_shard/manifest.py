from gguf_shard.validate import validate_shard_set, parse_shard_filename

def merge_manifests(shards_data):
    filenames = [s["filename"] for s in shards_data]
    valid, msg = validate_shard_set(filenames)
    if not valid:
        raise ValueError(f"Invalid shard set: {msg}")

    sorted_shards = sorted(shards_data, key=lambda x: parse_shard_filename(x["filename"])[1])

    unified_tensors = []
    total_size = 0
    for s in sorted_shards:
        for t in s.get("tensors", []):
            t_copy = dict(t)
            t_copy["shard_filename"] = s["filename"]
            unified_tensors.append(t_copy)
        total_size += s.get("file_size", 0)

    return {
        "prefix": parse_shard_filename(sorted_shards[0]["filename"])[0],
        "total_shards": len(sorted_shards),
        "tensors": unified_tensors,
        "total_size": total_size
    }
