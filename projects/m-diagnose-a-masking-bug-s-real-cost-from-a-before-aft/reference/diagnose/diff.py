def compute_diff(baseline, masked):
    diff = {}
    for k in baseline:
        if k in masked:
            b_val = baseline[k]
            m_val = masked[k]
            ratio = m_val / b_val if b_val != 0 else 0.0
            diff[k] = {"baseline": b_val, "masked": m_val, "ratio": ratio}
    return diff
