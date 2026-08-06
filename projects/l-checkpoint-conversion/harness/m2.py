import json
import os

import ref


def check(workdir):
    from convertkit import shards

    out = {"index_loaded": 0.0, "resolve_match": 0.0, "clean_index_ok": 0.0,
           "catches_missing": 0.0, "catches_size": 0.0}
    idx = shards.load_index(ref.index_path())
    with open(ref.index_path(), encoding="utf-8") as f:
        raw = json.load(f)
    if (idx.get("weight_map") == raw["weight_map"]
            and idx.get("total_size") == raw["metadata"]["total_size"]):
        out["index_loaded"] = 1.0

    resolved = shards.resolve(idx, ref.ST)
    tensors = resolved.get("tensors", {})
    if (sorted(tensors) == sorted(raw["weight_map"])
            and not resolved.get("missing")
            and all(tensors[k].get("shard") == v for k, v in raw["weight_map"].items())):
        out["resolve_match"] = 1.0

    if not shards.validate_index(idx, ref.ST):
        out["clean_index_ok"] = 1.0

    broken = {"total_size": idx["total_size"],
              "weight_map": dict(idx["weight_map"], ghost_tensor="model-00001-of-00002.safetensors")}
    if shards.validate_index(broken, ref.ST):
        out["catches_missing"] = 1.0

    wrong = {"total_size": idx["total_size"] + 4096,
             "weight_map": dict(idx["weight_map"])}
    if any("total" in str(p).lower() or "byte" in str(p).lower()
           for p in shards.validate_index(wrong, ref.ST)):
        out["catches_size"] = 1.0
    return out
