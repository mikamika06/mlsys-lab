import ref

def check(workdir):
    from blobgraph.graph import build_tag_graph, find_shared_tags
    from blobgraph.cp import simulate_cp

    out = {"graph_matched": 0.0, "cp_cost_zero": 0.0}

    want_graph = ref.build_tag_graph(ref.TAGS_MAPPING)
    try:
        got_graph = build_tag_graph(ref.TAGS_MAPPING)
    except Exception as e:
        out["_note"] = f"build_tag_graph raised {type(e).__name__}: {str(e)[:100]}"
        return out

    if got_graph == want_graph:
        out["graph_matched"] = 1.0
    else:
        out["_note"] = f"graph mismatch: got {got_graph}, want {want_graph}"
        return out

    try:
        shared = find_shared_tags(ref.TAGS_MAPPING, "sha256:layerA")
        want_shared = ref.find_shared_tags(ref.TAGS_MAPPING, "sha256:layerA")
        if sorted(shared) != sorted(want_shared):
            out["_note"] = f"find_shared_tags mismatch: got {shared}, want {want_shared}"
            return out
    except Exception as e:
        out["_note"] = f"find_shared_tags raised {type(e).__name__}"
        return out

    try:
        cost = simulate_cp(ref.TAGS_MAPPING, "model:latest", "model:backup")
        if cost == 0:
            out["cp_cost_zero"] = 1.0
        else:
            out["_note"] = f"simulate_cp cost non-zero: {cost}"
    except Exception as e:
        out["_note"] = f"simulate_cp raised {type(e).__name__}"

    return out
