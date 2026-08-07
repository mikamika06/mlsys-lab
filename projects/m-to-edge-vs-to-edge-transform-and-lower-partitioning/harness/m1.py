import ref

def check(workdir):
    from edge.export import to_edge_export
    out = {"partition_matched": 0.0}
    model = {"nodes": ["a", "b", "c", "d"]}
    try:
        res = to_edge_export(model, [1, 2])
        if isinstance(res, dict) and res.get("exported") is True:
            out["partition_matched"] = 1.0
    except Exception as e:
        out["_note"] = str(e)
    return out
