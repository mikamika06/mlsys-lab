from .names import map_index
from .shapes import is_quantised, target_shape

WIDTH = {"F32": 4, "F16": 2, "BF16": 2}


def conversion_plan(tensors, experts=0, out_dtype="F16"):
    width = WIDTH[out_dtype]
    mapping = map_index(tensors, experts)
    read_bytes = sum(t["n_bytes"] for t in tensors)
    write_bytes = 0
    dequantised = 0
    per_tensor = []
    for t in tensors:
        elems = 1
        for d in t["shape_ggml_order"]:
            elems *= d
        out = elems * width
        write_bytes += out
        if is_quantised(t["ggml_type"]):
            dequantised += 1
        per_tensor.append({
            "name": t["name"],
            "target": mapping["mapped"].get(t["name"])
            or mapping["fanned_out"].get(t["name"]),
            "target_shape": target_shape(t["shape_ggml_order"]),
            "source_bytes": t["n_bytes"],
            "output_bytes": out,
            "needs_dequantisation": is_quantised(t["ggml_type"]),
            "expansion": out / t["n_bytes"] if t["n_bytes"] else 0.0,
        })
    return {
        "out_dtype": out_dtype,
        "tensors": len(tensors),
        "unmapped": mapping["unmapped"],
        "fanned_out": len(mapping["fanned_out"]),
        "target_tensors": mapping["target_count"],
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "expansion": write_bytes / read_bytes if read_bytes else 0.0,
        "dequantised_tensors": dequantised,
        "per_tensor": per_tensor,
    }


def shard_plan(plan, shard_bytes):
    """Split the output into shards no larger than shard_bytes, keeping order."""
    shards, cur, size = [], [], 0
    for t in plan["per_tensor"]:
        if cur and size + t["output_bytes"] > shard_bytes:
            shards.append({"tensors": cur, "bytes": size})
            cur, size = [], 0
        cur.append(t["name"])
        size += t["output_bytes"]
    if cur:
        shards.append({"tensors": cur, "bytes": size})
    for i, s in enumerate(shards, 1):
        s["file"] = "model-%05d-of-%05d.safetensors" % (i, len(shards))
    return shards
