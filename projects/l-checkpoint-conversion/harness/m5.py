import ref


def check(workdir):
    from convertkit import plan

    out = {"bytes_match": 0.0, "expansion_match": 0.0, "dequant_count": 0.0,
           "shards_cover": 0.0, "shard_limit": 0.0}

    doc = ref.gguf_index("llama")
    want = ref.expect_plan("llama")
    got = plan.conversion_plan(doc["tensors"], out_dtype="F16")
    if (got.get("read_bytes") == want["read_bytes"]
            and got.get("write_bytes") == want["write_bytes"]):
        out["bytes_match"] = 1.0
    if ref.near(got.get("expansion", -1), want["expansion"], 1e-9):
        out["expansion_match"] = 1.0
    if got.get("dequantised_tensors") == want["dequantised"]:
        out["dequant_count"] = 1.0

    limit = 4 * 1024 ** 3
    sh = plan.shard_plan(got, limit)
    covered = [n for s in sh for n in s["tensors"]]
    if len(covered) == len(doc["tensors"]) and len(set(covered)) == len(covered):
        out["shards_cover"] = 1.0
    if sh and all(s["bytes"] <= limit or len(s["tensors"]) == 1 for s in sh):
        out["shard_limit"] = 1.0
    return out
