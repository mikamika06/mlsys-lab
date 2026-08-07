import numpy as np
import ref


def check(workdir):
    from irconv.parity import compute_relative_error, evaluate_ir_node, verify_conversion_parity

    out = {"parity_matched": 0.0, "rel_err_valid": 0.0}

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.1], dtype=np.float32)
    err = compute_relative_error(a, b)
    if 0.02 < err < 0.03:
        out["rel_err_valid"] = 1.0

    pt_outputs, ir_graph = ref.generate_parity_data()
    res = verify_conversion_parity(pt_outputs, ir_graph)

    if "Y" in res and "h1" in res and res["Y"] < 1e-3 and res["h1"] < 1e-3:
        out["parity_matched"] = 1.0
    else:
        out["_note"] = f"Parity output mismatch: {res}"

    return out
