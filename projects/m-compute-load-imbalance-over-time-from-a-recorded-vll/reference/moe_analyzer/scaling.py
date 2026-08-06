import numpy as np


def fit_and_extrapolate_throughput(benchmark_rows, target_ep_degrees):
    ep_degrees = np.array([r["ep_degree"] for r in benchmark_rows], dtype=np.float64)
    throughputs = np.array([r["tokens_per_sec"] for r in benchmark_rows], dtype=np.float64)

    log_ep = np.log(ep_degrees)
    log_tp = np.log(throughputs)

    poly = np.polyfit(log_ep, log_tp, 1)
    scaling_exponent = float(poly[0])
    intercept = float(poly[1])

    extrapolated = {}
    for ep in target_ep_degrees:
        pred_log_tp = scaling_exponent * np.log(float(ep)) + intercept
        extrapolated[int(ep)] = float(np.exp(pred_log_tp))

    return {
        "scaling_exponent": scaling_exponent,
        "intercept": intercept,
        "extrapolated_throughput": extrapolated,
    }
