import json
import os

from .safetensors import entries


def load_index(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return {"total_size": doc.get("metadata", {}).get("total_size", 0),
            "weight_map": dict(doc.get("weight_map", {}))}


def resolve(index, directory):
    """Every tensor in the index, with the shard and byte range it lives in."""
    cache = {}
    out = {}
    missing = []
    for name, shard in sorted(index["weight_map"].items()):
        path = os.path.join(directory, shard)
        if shard not in cache:
            if not os.path.isfile(path):
                missing.append(shard)
                cache[shard] = None
            else:
                with open(path, "rb") as f:
                    cache[shard] = entries(f.read())
        parsed = cache[shard]
        if parsed is None:
            continue
        found = next((t for t in parsed["tensors"] if t["name"] == name), None)
        if found is None:
            missing.append("%s in %s" % (name, shard))
            continue
        out[name] = dict(found, shard=shard)
    return {"tensors": out, "missing": missing}


def validate_index(index, directory):
    problems = []
    resolved = resolve(index, directory)
    for m in resolved["missing"]:
        problems.append("index names %s but it is not there" % m)

    shards = sorted(set(index["weight_map"].values()))
    total = 0
    for shard in shards:
        path = os.path.join(directory, shard)
        if os.path.isfile(path):
            total += os.path.getsize(path)
    if index["total_size"] and total != index["total_size"]:
        problems.append("index totals %d bytes, the shards total %d"
                        % (index["total_size"], total))

    for shard in shards:
        path = os.path.join(directory, shard)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            parsed = entries(f.read())
        for t in parsed["tensors"]:
            if index["weight_map"].get(t["name"]) != shard:
                problems.append("%s is in %s but the index does not say so"
                                % (t["name"], shard))
    return problems
