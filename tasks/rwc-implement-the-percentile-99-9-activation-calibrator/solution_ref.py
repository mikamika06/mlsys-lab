import numpy as np


def percentile_amax(calibration_batches):
    values = np.concatenate(
        [np.asarray(batch).reshape(-1) for batch in calibration_batches]
    )
    return float(np.percentile(np.abs(values), 99.9))
