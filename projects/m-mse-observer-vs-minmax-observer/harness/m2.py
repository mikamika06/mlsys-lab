import ref
import numpy as np


def check(workdir):
    from quant.observer import MinMaxObserver, MSEObserver

    out = {"mse_better": 0.0, "minmax_match": 0.0}
    data = ref.DATASETS[0]

    mm = MinMaxObserver(bits=8, symmetric=True)
    mm.update(data)
    scale_mm, zp_mm = mm.compute_params()
    q_mm = np.clip(np.round(data / scale_mm), -128, 127) * scale_mm
    mse_mm = np.mean((data - q_mm) ** 2)

    mse_obs = MSEObserver(bits=8, symmetric=True, num_bins=20)
    mse_obs.update(data)
    scale_mse, zp_mse = mse_obs.compute_params()
    q_mse = np.clip(np.round(data / scale_mse), -128, 127) * scale_mse
    mse_val = np.mean((data - q_mse) ** 2)

    if zp_mm == 0 and zp_mse == 0:
        out["minmax_match"] = 1.0

    if mse_val <= mse_mm:
        out["mse_better"] = 1.0

    return out
