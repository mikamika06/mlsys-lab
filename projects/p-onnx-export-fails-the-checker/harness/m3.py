import ref
import os

def check(workdir):
    from exporter.fixer import patch_custom_layer
    path = os.path.join(workdir, "model.onnx")
    res = patch_custom_layer(path)
    return {"nodes_replaced": float(res)}
