import numpy as np


def verify_prefill_update(model_outputs, reference_cache):
    errs = [np.max(np.abs(np.array(mo) - np.array(rc))) for mo, rc in zip(model_outputs, reference_cache)]
    return float(max(errs))
