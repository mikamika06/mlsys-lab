import os
import tempfile
import numpy as np
import tvm
from tvm.relax.frontend.onnx import from_onnx
import onnx


def roundtrip_module_test(onnx_bytes: bytes, input_data: np.ndarray) -> bool:
    model = onnx.load_from_string(onnx_bytes)
    mod_relax = from_onnx(model)
    target = tvm.target.Target("llvm", host="llvm")
    with tvm.transform.PassContext(opt_level=2):
        ex = tvm.compile(mod_relax, target=target)

    dev = tvm.cpu(0)
    vm = tvm.relax.VirtualMachine(ex, dev)
    out_orig = vm["main"](tvm.nd.array(input_data)).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        lib_path = os.path.join(tmpdir, "model.so")
        ex.export_library(lib_path)
        loaded_lib = tvm.runtime.load_module(lib_path)
        vm_loaded = tvm.relax.VirtualMachine(loaded_lib, dev)
        out_loaded = vm_loaded["main"](tvm.nd.array(input_data)).numpy()

    return bool(np.allclose(out_orig, out_loaded, atol=1e-5))


def measure_artifact_sizes(onnx_bytes: bytes) -> dict:
    model = onnx.load_from_string(onnx_bytes)
    mod_relax = from_onnx(model)
    target = tvm.target.Target("llvm", host="llvm")
    sizes = {}

    for level in [0, 2, 3]:
        with tvm.transform.PassContext(opt_level=level):
            ex = tvm.compile(mod_relax, target=target)
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_path = os.path.join(tmpdir, f"model_opt{level}.so")
            ex.export_library(lib_path)
            size = os.path.getsize(lib_path)
            sizes[level] = int(size)

    return sizes
