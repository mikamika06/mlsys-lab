import numpy as np


def compute_goodput(traces, ttft_slo, tpot_slo):
    count = 0
    for tr in traces:
        if tr["ttft"] <= ttft_slo and tr["tpot"] <= tpot_slo:
            count += 1
    return float(count) / float(len(traces)) if traces else 0.0


def compute_e2el_gap(traces):
    gaps = []
    for tr in traces:
        ts = tr["timestamps"]
        actual = ts[-1] - ts[0]
        est = tr["ttft"] + tr["tpot"] * max(0, tr["num_tokens"] - 1)
        gaps.append(actual - est)
    return float(np.mean(np.abs(gaps)))


def compute_itl_and_tpot(timestamps):
    if len(timestamps) < 3:
        return 0.0, 0.0
    itls = [timestamps[i] - timestamps[i-1] for i in range(2, len(timestamps))]
    tpot = float(np.mean(itls))
    max_itl = float(np.max(itls))
    return tpot, max_itl
