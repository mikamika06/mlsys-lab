import ref
import numpy as np

def check(workdir):
    from mlxops.device import get_active_device, execute_op, safe_float64_exec

    out = {
        "gpu_exception_caught": 0.0,
        "cpu_workaround_correct": 0.0,
        "stream_state_restored": 0.0,
    }

    f64_tensor = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    dummy_op = lambda x: x * 2.0

    caught = False
    try:
        execute_op(dummy_op, f64_tensor)
    except ValueError as e:
        if "float64" in str(e):
            caught = True

    if caught:
        out["gpu_exception_caught"] = 1.0
    else:
        out["_note"] = "execute_op on GPU did not raise ValueError for float64"

    try:
        res = safe_float64_exec(dummy_op, f64_tensor)
        expected = ref.safe_float64_exec(dummy_op, f64_tensor)
        if np.allclose(res, expected):
            out["cpu_workaround_correct"] = 1.0
        else:
            out["_note"] = f"Result mismatch: got {res}, expected {expected}"
    except Exception as e:
        out["_note"] = f"safe_float64_exec raised error: {type(e).__name__}: {e}"

    if get_active_device() == "gpu":
        out["stream_state_restored"] = 1.0
    else:
        out["_note"] = f"Active device after exec was {get_active_device()}, expected gpu"

    return out
