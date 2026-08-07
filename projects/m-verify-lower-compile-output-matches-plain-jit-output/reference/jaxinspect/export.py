import numpy as np


def verify_serialized_numerics(
    original_outputs: list[dict],
    deserialized_outputs: list[dict],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> dict:
    all_matched = True
    max_err = 0.0
    for orig, des in zip(original_outputs, deserialized_outputs):
        a = np.asarray(orig["data"])
        b = np.asarray(des["data"])
        err = float(np.max(np.abs(a - b)))
        if err > max_err:
            max_err = err
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            all_matched = False
    return {
        "max_abs_err": max_err,
        "matches": all_matched,
    }
