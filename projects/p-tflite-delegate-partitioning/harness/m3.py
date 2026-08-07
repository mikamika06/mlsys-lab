import os
import ref

def check(workdir):
    m = {"fused_subgraphs": 0.0}
    path = ref.create_dummy_model(workdir)
    out_path = os.path.join(workdir, "optimized.tflite")
    try:
        from edge.partitioner import rewrite_graph
        res = rewrite_graph(path, out_path)
        if res and os.path.exists(out_path):
            m["fused_subgraphs"] = 1.0
    except Exception:
        pass
    return m
