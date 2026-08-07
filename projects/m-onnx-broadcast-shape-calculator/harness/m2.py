import ref


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from onnxcalc.value_info import infer_graph_value_info
    except Exception as e:
        return {"graphs_rebuilt": 0.0, "total": float(len(ref.TEST_GRAPHS)), "_note": f"import failed: {e}"}

    out = {"graphs_rebuilt": 0.0, "total": float(len(ref.TEST_GRAPHS))}
    ok = 0
    for idx, (graph, want) in enumerate(ref.TEST_GRAPHS):
        if want is ValueError or (isinstance(want, type) and issubclass(want, Exception)):
            try:
                infer_graph_value_info(graph)
                if "_note" not in out:
                    out["_note"] = f"graph {idx}: expected ValueError, but succeeded"
            except ValueError:
                ok += 1
            except Exception as e:
                if "_note" not in out:
                    out["_note"] = f"graph {idx}: expected ValueError, got {type(e).__name__}"
        else:
            try:
                got = infer_graph_value_info(graph)
                if got == want:
                    ok += 1
                elif "_note" not in out:
                    out["_note"] = f"graph {idx}: got {got}, want {want}"
            except Exception as e:
                if "_note" not in out:
                    out["_note"] = f"graph {idx}: raised {type(e).__name__}: {str(e)}"
    out["graphs_rebuilt"] = float(ok)
    return out
