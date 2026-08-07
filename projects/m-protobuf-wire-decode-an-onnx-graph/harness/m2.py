import ref


def check(workdir):
    from onnxdecode.initializers import dump_initializers
    ok = 0
    out = {"manifests_matched": 0.0}
    for i, (_, expected_graph) in enumerate(ref.SAMPLE_GRAPHS):
        try:
            manifests = dump_initializers(expected_graph)
            if len(manifests) == len(expected_graph["initializers"]):
                match = True
                for m, init in zip(manifests, expected_graph["initializers"]):
                    if m.get("name") != init["name"] or m.get("size_bytes") != len(init["raw_data"]):
                        match = False
                if match:
                    ok += 1
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"manifest {i} failed: {type(e).__name__}: {str(e)[:100]}"
    out["manifests_matched"] = float(ok)
    return out
