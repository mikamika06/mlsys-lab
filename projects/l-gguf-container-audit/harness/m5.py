import ref


def check(workdir):
    from ggufkit import plan

    blob = ref.clean_blob()
    want = ref.truth()
    out = {"split_match": 0.0, "pages_match": 0.0, "subset_smaller": 0.0,
           "waste_match": 0.0}

    p = plan.load_plan(blob, page=16384)
    weight_bytes = sum(t["n_bytes"] for t in want["tensors"])
    if (p.get("metadata_bytes") == want["tensors"][0]["absolute_data_offset"]
            - want["tensors"][0].get("offset_from_data_start", 0)
            and p.get("weight_bytes") == weight_bytes
            and p.get("file_bytes") == want["file_bytes"]):
        out["split_match"] = 1.0

    pages = set()
    for t in want["tensors"]:
        s = t["absolute_data_offset"]
        e = s + t["n_bytes"]
        for pg in range(s // 16384, (e - 1) // 16384 + 1):
            pages.add(pg)
    if p.get("distinct_pages") == len(pages) and p.get("resident_bytes") == len(pages) * 16384:
        out["pages_match"] = 1.0

    one = want["tensors"][0]["name"]
    sub = plan.load_plan(blob, page=16384, want=[one])
    if 0 < sub["resident_bytes"] < p["resident_bytes"]:
        out["subset_smaller"] = 1.0

    rows = {r["name"]: r for r in p.get("tensors", [])}
    ok = 1.0
    for t in want["tensors"]:
        r = rows.get(t["name"])
        if not r:
            ok = 0.0
            continue
        s = t["absolute_data_offset"]
        e = s + t["n_bytes"]
        span = (e - 1) // 16384 - s // 16384 + 1
        if r.get("pages") != span or r.get("waste_bytes") != span * 16384 - (e - s):
            ok = 0.0
    out["waste_match"] = ok
    return out
