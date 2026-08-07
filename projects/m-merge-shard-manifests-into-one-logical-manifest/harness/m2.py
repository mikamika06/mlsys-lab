import ref


def check(workdir):
    from shards.manifest import merge_manifests
    from shards.validate import check_shard_set

    case = ref.get_test_cases()
    manifests = case["manifests"]
    filenames = ["model-0001-of-0002.gguf", "model-0002-of-0002.gguf"]

    out = {"manifests_merged": 0.0, "tensors_correct": 0.0}
    try:
        merged = merge_manifests(manifests)
        if merged.get("total_size") == 3072 and "weight_a" in merged.get("tensors", {}) and "weight_b" in merged.get("tensors", {}):
            out["manifests_merged"] = 1.0
            out["tensors_correct"] = 1.0
        else:
            out["_note"] = f"Merged output incorrect: {merged}"
    except Exception as e:
        out["_note"] = f"merge_manifests raised exception: {type(e).__name__}: {e}"
    return out
