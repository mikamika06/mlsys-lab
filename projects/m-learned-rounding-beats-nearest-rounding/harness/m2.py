import ref
import numpy as np


def check(workdir):
    from quantlib.autoround import AutoRoundModifier

    model = ref.get_tiny_model()
    modifier = AutoRoundModifier(model)
    calib_data = [np.zeros((16, 16))]

    try:
        res = modifier.optimize(calib_data)
    except Exception as e:
        return {"autoround_converged": 0.0, "_note": f"optimize raised {type(e).__name__}: {str(e)[:100]}"}

    if not isinstance(res, list) or len(res) != len(model["weights"]):
        return {"autoround_converged": 0.0, "_note": "Invalid output shape or type from AutoRoundModifier"}

    orig_mse = float(np.mean([(w - o) ** 2 for w, o in zip(model["weights"], res)]))
    out = {"autoround_converged": 1.0, "mse": orig_mse}
    return out
