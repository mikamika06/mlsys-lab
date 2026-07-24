def compute_peak_activation_bytes(schedule, M, p, a_bytes, d):
    if schedule == "gpipe":
        return M * a_bytes * d
    elif schedule == "1f1b":
        return (p - 1) * a_bytes * d
    else:
        raise ValueError("schedule must be 'gpipe' or '1f1b'")
