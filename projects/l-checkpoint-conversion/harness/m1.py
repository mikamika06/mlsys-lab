import ref


def check(workdir):
    from convertkit import safetensors as S

    out = {"entries_match": 1.0, "offsets_match": 1.0, "metadata_match": 0.0,
           "clean_accepted": 0.0, "damaged_rejected": 0.0, "damaged_named": 0.0}
    for name in ref.shard_names():
        body = ref.blob(name)
        got = S.entries(body)
        want = ref.expect_entries(name)
        if got.get("data_start") != want["data_start"]:
            out["offsets_match"] = 0.0
        g = {t["name"]: t for t in got["tensors"]}
        if sorted(g) != sorted(t["name"] for t in want["tensors"]):
            out["entries_match"] = 0.0
            continue
        for w in want["tensors"]:
            t = g[w["name"]]
            if (t.get("dtype") != w["dtype"] or list(t.get("shape", [])) != w["shape"]
                    or t.get("elements") != w["elements"]
                    or t.get("expected_bytes") != w["expected_bytes"]):
                out["entries_match"] = 0.0
            if list(t.get("absolute_offsets", [])) != w["absolute_offsets"]:
                out["offsets_match"] = 0.0
        if got.get("metadata", {}).get("format") == "np":
            out["metadata_match"] = 1.0
        if not S.validate(body):
            out["clean_accepted"] = 1.0

    dmg = ref.damage_truth()
    problems = S.validate(ref.blob(dmg["file"]))
    if problems:
        out["damaged_rejected"] = 1.0
        if any(dmg["tensor"] in str(p) for p in problems):
            out["damaged_named"] = 1.0
    return out
