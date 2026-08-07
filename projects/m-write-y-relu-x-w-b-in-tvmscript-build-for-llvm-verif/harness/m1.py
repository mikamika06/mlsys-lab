import numpy as np
import ref


def check(workdir):
    from tvm_pipeline.ops import run_llvm_relu_matmul_add, build_relu_matmul_add_script

    out = {"max_abs_err": 1000.0}
    try:
        script_code = build_relu_matmul_add_script(4, 6, 8)
        if not isinstance(script_code, str) or "R.nn.relu" not in script_code:
            out["_note"] = "build_relu_matmul_add_script did not return expected TVMScript string"
            return out
    except Exception as e:
        out["_note"] = f"build_relu_matmul_add_script raised exception: {e}"
        return out

    x, w, b = ref.get_test_tensors()
    want = ref.compute_ref_relu_matmul_add(x, w, b)
    try:
        got = run_llvm_relu_matmul_add(x, w, b)
        err = float(np.max(np.abs(got - want)))
        out["max_abs_err"] = err
    except Exception as e:
        out["_note"] = f"run_llvm_relu_matmul_add failed: {e}"

    return out
