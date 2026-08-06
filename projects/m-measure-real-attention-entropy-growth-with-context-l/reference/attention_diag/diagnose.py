import numpy as np


def diagnose_niah_failure(entropy_profile, rope_scaling_factor):
    mean_entropy = float(np.mean(entropy_profile))
    max_entropy = float(np.max(entropy_profile))
    if rope_scaling_factor < 1.0 and mean_entropy > 4.5:
        return "rope_extrapolation_failure"
    elif max_entropy > 6.0:
        return "attention_dilution"
    else:
        return "stable_attention"
