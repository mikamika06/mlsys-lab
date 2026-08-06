import ref

def check(workdir):
    from edge_export.census import compute_census
    out = {"census_match": 0.0}
    sample_nodes = [
        {"target": "xnnpack"},
        {"target": "xnnpack"},
        {"target": "cpu_fallback"}
    ]
    got = compute_census(sample_nodes)
    want = {"delegated": 2, "fallback": 1, "ratio": 2.0 / 3.0}
    if got.get("delegated") == want["delegated"] and got.get("fallback") == want["fallback"] and abs(got.get("ratio", 0) - want["ratio"]) < 1e-5:
        out["census_match"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
