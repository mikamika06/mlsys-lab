import ref


def check(workdir):
    from ggufkit import container

    blob = ref.clean_blob()
    want = ref.truth()
    out = {"header_match": 0.0, "kv_count_match": 0.0,
           "kv_values_match": 0.0, "kv_types_match": 0.0}

    head = container.parse_header(blob)
    if (head.get("magic") == want["magic"] and head.get("version") == want["version"]
            and head.get("tensor_count") == want["tensor_count"]
            and head.get("kv_count") == want["kv_count"]):
        out["header_match"] = 1.0

    meta = container.parse_kv(blob)
    kv = meta["kv"] if isinstance(meta, dict) and "kv" in meta else meta
    if len(kv) == want["kv_count"]:
        out["kv_count_match"] = 1.0

    values_ok, types_ok = 1.0, 1.0
    types = meta.get("types", {}) if isinstance(meta, dict) else {}
    for key, rec in want["kv"].items():
        if key not in kv:
            values_ok = 0.0
            continue
        got, expect = kv[key], rec["value"]
        if isinstance(expect, list):
            got_list = list(got) if isinstance(got, (list, tuple)) else None
            if got_list is None or got_list[:len(expect)] != expect:
                values_ok = 0.0
        elif isinstance(expect, float):
            if not isinstance(got, (int, float)) or abs(got - expect) > 1e-6 * max(1.0, abs(expect)):
                values_ok = 0.0
        elif got != expect:
            values_ok = 0.0
        name = types.get(key, "")
        if rec["type"] == "ARRAY":
            if not name.startswith("ARRAY"):
                types_ok = 0.0
        elif name != rec["type"]:
            types_ok = 0.0
    out["kv_values_match"] = values_ok
    out["kv_types_match"] = types_ok
    return out
