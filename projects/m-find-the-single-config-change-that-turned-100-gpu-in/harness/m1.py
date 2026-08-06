import ref

def check(workdir):
    from offload.finder import find_config_change
    out = {"found_match": 0.0}
    c1 = {"a": 1, "b": 2, "c": 3}
    c2 = {"a": 1, "b": 99, "c": 3}
    got = find_config_change(c1, c2)
    want = ref.find_config_change(c1, c2)
    if got == want:
        out["found_match"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
