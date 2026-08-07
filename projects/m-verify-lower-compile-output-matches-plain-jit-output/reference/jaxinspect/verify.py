import numpy as np


def verify_compile_vs_jit(aot_outputs: list[dict], jit_outputs: list[dict]) -> dict:
    max_err = 0.0
    for aot_item, jit_item in zip(aot_outputs, jit_outputs):
        aot_arr = np.asarray(aot_item["data"])
        jit_arr = np.asarray(jit_item["data"])
        err = float(np.max(np.abs(aot_arr - jit_arr)))
        if err > max_err:
            max_err = err
    return {
        "max_abs_err": max_err,
        "is_close": bool(max_err <= 1e-5),
    }
