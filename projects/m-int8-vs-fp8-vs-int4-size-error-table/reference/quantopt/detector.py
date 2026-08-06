import numpy as np


def detect_and_clean_outliers(calib_samples, z_threshold=4.0):
    cleaned = []
    poison_flags = []

    for sample in calib_samples:
        s_max = float(np.max(np.abs(sample)))
        s_mean = float(np.mean(sample))
        s_std = float(np.std(sample))

        if s_std == 0.0:
            z_score = 0.0
        else:
            z_score = (s_max - abs(s_mean)) / s_std

        is_poison = z_score > z_threshold
        poison_flags.append(bool(is_poison))

        if is_poison:
            p99 = float(np.percentile(np.abs(sample), 99.0))
            cleaned_sample = np.clip(sample, -p99, p99)
            cleaned.append(cleaned_sample)
        else:
            cleaned.append(sample.copy())

    return {
        "cleaned_samples": cleaned,
        "poison_flags": poison_flags,
        "num_poisoned": int(sum(poison_flags)),
    }
