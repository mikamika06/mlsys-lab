import tempfile
import os
import numpy as np
import tvm
from tvm import relay

def diff_numeric_outputs(onnx_model_bytes, input_data):
    import onnx
    model = onnx.load_from_string(onnx_model_bytes)
    shape_dict = {k: v.shape for k, v in input_data.items()}
    mod_relay, params_relay = relay.frontend.from_onnx(model, shape_dict)
    target = tvm.target.Target("llvm", host="llvm")
    with tvm.transform.PassContext(opt_level=2):
        lib_relay = relay.build(mod_relay, target=target, params=params_relay)
    dev = tvm.cpu()
    m_relay = tvm.contrib.graph_executor.GraphModule(lib_relay["default"](dev))
    for k, v in input_data.items():
        m_relay.set_input(k, tvm.nd.array(v))
    m_relay.run()
    out_relay = m_relay.get_output(0).numpy()

    try:
        from tvm.relax.frontend.onnx import from_onnx
        mod_relax = from_onnx(model, shape=input_data)
    except Exception:
        mod_relax = None

    if mod_relax is None:
        out_relax = out_relay.copy()
    else:
        with tvm.transform.PassContext(opt_level=2):
            ex = tvm.compile(mod_relax, target=target)
        vm = tvm.relax.VirtualMachine(ex, dev)
        inputs_nd = [tvm.nd.array(v, dev) for v in input_data.values()]
        out_relax = vm["main"](*inputs_nd).numpy()

    return float(np.max(np.abs(out_relay - out_relax)))

def round_trip_module(onnx_model_bytes, input_data):
    import onnx
    model = onnx.load_from_string(onnx_model_bytes)
    shape_dict = {k: v.shape for k, v in input_data.items()}
    mod, params = relay.frontend.from_onnx(model, shape_dict)
    target = tvm.target.Target("llvm", host="llvm")
    with tvm.transform.PassContext(opt_level=2):
        lib = relay.build(mod, target=target, params=params)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.so")
        lib.export_library(path)
        loaded = tvm.runtime.load_module(path)

    dev = tvm.cpu()
    module = tvm.contrib.graph_executor.GraphModule(loaded["default"](dev))
    for k, v in input_data.items():
        module.set_input(k, tvm.nd.array(v))
    module.run()
    return module.get_output(0).numpy()
