import ref


def check(workdir):
    from ggufkit import container

    blob = ref.clean_blob()
    want = ref.truth()
    out = {"index_match": 0.0, "alignment_match": 0.0, "offsets_match": 0.0,
           "clean_accepted": 0.0, "corrupt_rejected": 0.0, "names_the_tensor": 0.0}

    index = container.parse_tensor_index(blob)
    tensors = index["tensors"] if isinstance(index, dict) else index
    by_name = {t["name"]: t for t in tensors}
    if len(tensors) == want["tensor_count"] and set(by_name) == {
            t["name"] for t in want["tensors"]}:
        out["index_match"] = 1.0
    if isinstance(index, dict) and index.get("alignment") == want["alignment"]:
        out["alignment_match"] = 1.0

    offsets_ok, shapes_ok = 1.0, True
    for t in want["tensors"]:
        got = by_name.get(t["name"])
        if not got:
            offsets_ok = 0.0
            continue
        if got.get("absolute_data_offset") != t["absolute_data_offset"]:
            offsets_ok = 0.0
        if (list(got.get("shape_ggml_order", [])) != t["shape_ggml_order"]
                or got.get("n_bytes") != t["n_bytes"]
                or got.get("ggml_type_id") != t["ggml_type_id"]):
            shapes_ok = False
    out["offsets_match"] = offsets_ok if shapes_ok else 0.0

    if not container.validate(blob):
        out["clean_accepted"] = 1.0
    problems = container.validate(ref.corrupt_blob())
    if problems:
        out["corrupt_rejected"] = 1.0
        if any(ref.corruption()["damaged_tensor"] in str(p) for p in problems):
            out["names_the_tensor"] = 1.0
    return out
