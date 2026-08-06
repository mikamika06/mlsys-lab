import tempfile
import os
import tvm
from tvm import relay

def measure_artifact_sizes(onnx_model_bytes, input_data):
    import onnx
    model = onnx.load_from_string(onnx_model_bytes)
    shape_dict = {k: v.shape for k, v in input_data.items()}
    mod, params = relay.frontend.from_onnx(model, shape_dict)
    target = tvm.target.Target("llvm", host="llvm")

    sizes = {}
    for opt in [0, 2, 3]:
        with tvm.transform.PassContext(opt_level=opt):
            lib = relay.build(mod, target=target, params=params)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"model_{opt}.so")
            lib.export_library(path)
            sizes[opt] = os.path.getsize(path)
    return sizes
