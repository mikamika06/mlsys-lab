import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    import transform
    import ref

    out = {"graph_structure": 0.0, "max_abs_err": 999.0}
    graph = ref.make_graph()
    x = ref.make_input()

    try:
        truth_tensor, _ = ref.run_and_isolate(graph, x, np.float32)
        new_graph = transform.insert_cast_nodes(graph, "exp_act", "norm1")

        has_f32 = any(L.get("op") == "cast" and L.get("to") == "float32" for L in new_graph)
        has_f16 = any(L.get("op") == "cast" and L.get("to") == "float16" for L in new_graph)
        if has_f32 and has_f16:
            out["graph_structure"] = 1.0

        got_tensor, got_invalid = ref.run_and_isolate(new_graph, x, np.float16)

        if got_invalid is None:
            err = np.max(np.abs(truth_tensor - got_tensor))
            out["max_abs_err"] = float(err)
        else:
            out["_note"] = f"Still got invalid at {got_invalid}"

    except Exception as e:
        out["_note"] = f"Error during execution: {e}"

    return out
