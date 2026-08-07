import ref


def check(workdir):
    from onnxdecode.parser import decode_onnx_graph
    ok = 0
    out = {"graphs_matched": 0.0}
    for i, (raw, expected) in enumerate(ref.SAMPLE_GRAPHS):
        try:
            got = decode_onnx_graph(raw)
            if got.get("name") == expected["name"] and len(got.get("nodes", [])) == len(expected["nodes"]):
                ok += 1
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"graph {i} failed: {type(e).__name__}: {str(e)[:100]}"
    out["graphs_matched"] = float(ok)
    return out
