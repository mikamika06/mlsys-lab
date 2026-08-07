import ref
import os

def check(workdir):
    from exporter.fixer import fix_shapes
    path = os.path.join(workdir, "model.onnx")
    res = fix_shapes(path)
    return {"shapes_valid": float(res)}
