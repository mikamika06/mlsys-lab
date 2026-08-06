import ref
import numpy as np


def check(workdir):
    from pt2ex.quant import compute_qparams, convert_tensor
    _, weight = ref.get_test_fixtures()
    qp_tensor = compute_qparams(weight, per_channel=False)
    qp_channel = compute_qparams(weight, per_channel=True, axis=0)
    res_tensor = convert_tensor(weight, qp_tensor)
    res_channel = convert_tensor(weight, qp_channel)
    err_tensor = float(np.max(np.abs(weight - res_tensor)))
    err_channel = float(np.max(np.abs(weight - res_channel)))
    max_err = min(err_tensor, err_channel)
    return {"max_abs_err": max_err}
