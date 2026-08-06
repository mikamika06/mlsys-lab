import numpy as np
import tvm
from tvm import relay
from tvm.relax.frontend.onnx import from_onnx
import onnx


def compare_relay_relax_onnx(onnx_bytes: bytes, input_data: np.ndarray) -> float:
    model = onnx.load_from_string(onnx_bytes)

    shape_dict = {"input0": input_data.shape}
    mod_relay, params_relay = relay.frontend.from_onnx(model, shape_dict)
    target = tvm.target.Target("llvm", host="llvm")
    with tvm.transform.PassContext(opt_level=2):
        lib_relay = relay.build(mod_relay, target=target, params=params_relay)

    dev = tvm.cpu(0)
    m_relay = tvm.contrib.graph_executor.GraphModule(lib_relay["default"](dev))
    m_relay.set_input("input0", tvm.nd.array(input_data))
    m_relay.run()
    out_relay = m_relay.get_output(0).numpy()

    mod_relax = from_onnx(model, keep_stritt_params=True)
    with tvm.transform.PassContext(opt_level=2):
        ex = tvm.compile(mod_relax, target=target)
    vm = tvm.relax.VirtualMachine(ex, dev)
    out_relax = vm["main"](tvm.nd.array(input_data)).numpy()

    diff = np.max(np.abs(out_relay - out_relax))
    return float(diff)
