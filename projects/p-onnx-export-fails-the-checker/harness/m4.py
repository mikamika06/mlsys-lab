import ref
import os

def check(workdir):
    from exporter.optimizer import simplify_graph
    path = os.path.join(workdir, "model.onnx")
    res = simplify_graph(path)
    return {"graph_simplified": float(res)}
