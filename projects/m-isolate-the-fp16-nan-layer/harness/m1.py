import sys
import os
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    import engine
    import ref

    out = {"invalid_layer_matches": 0.0, "tensor_matches": 0.0}
    graph = ref.make_graph()
    x = ref.make_input()

    try:
        got_t16, got_inv16 = engine.run_and_isolate(graph, x, np.float16)
        want_t16, want_inv16 = ref.run_and_isolate(graph, x, np.float16)
        if got_inv16 == want_inv16:
            out["invalid_layer_matches"] = 1.0

        got_t32, got_inv32 = engine.run_and_isolate(graph, x, np.float32)
        want_t32, want_inv32 = ref.run_and_isolate(graph, x, np.float32)

        if got_inv32 is None and np.allclose(got_t32, want_t32, atol=1e-4):
            out["tensor_matches"] = 1.0

    except Exception as e:
        out["_note"] = f"Error during execution: {e}"

    return out
