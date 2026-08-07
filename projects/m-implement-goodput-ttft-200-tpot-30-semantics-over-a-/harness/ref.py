import numpy as np


def generate_traces():
    np.random.seed(42)
    traces = []
    for i in range(50):
        n_tokens = int(np.random.randint(10, 50))
        ttft = float(np.random.uniform(50, 350))
        tpot = float(np.random.uniform(10, 60))
        preempted = np.random.rand() < 0.3
        if preempted:
            pause = float(np.random.uniform(100, 500))
            ts = [0.0, ttft]
            current = ttft
            for t_idx in range(1, n_tokens):
                if t_idx == n_tokens // 2:
                    current += pause
                current += tpot + np.random.uniform(-2, 2)
                ts.append(current)
        else:
            ts = [0.0, ttft]
            current = ttft
            for t_idx in range(1, n_tokens):
                current += tpot + np.random.uniform(-2, 2)
                ts.append(current)

        actual_ttft = ts[1] - ts[0]
        if n_tokens > 1:
            itls = [ts[j] - ts[j-1] for j in range(2, len(ts))]
            actual_tpot = float(np.mean(itls)) if itls else 0.0
        else:
            actual_tpot = 0.0

        traces.append({
            "id": i,
            "ttft": actual_ttft,
            "tpot": actual_tpot,
            "num_tokens": n_tokens,
            "timestamps": ts,
            "preempted": preempted
        })
    return traces


def compute_goodput(traces, ttft_slo, tpot_slo):
    good = 0
    for t in traces:
        if t["ttft"] <= ttft_slo and t["tpot"] <= tpot_slo:
            good += 1
    return float(good) / float(len(traces)) if traces else 0.0


def compute_e2el_gap(traces):
    gaps = []
    for t in traces:
        e2el_actual = t["timestamps"][-1] - t["timestamps"][0]
        e2el_est = t["ttft"] + t["tpot"] * max(0, t["num_tokens"] - 1)
        gaps.append(abs(e2el_actual - e2el_est))
    return float(np.mean(gaps))


def compute_itl_tpot(timestamps):
    if len(timestamps) < 3:
        return 0.0, 0.0
    itls = [timestamps[i] - timestamps[i-1] for i in range(2, len(timestamps))]
    tpot = float(np.mean(itls)) if itls else 0.0
    max_itl = float(np.max(itls)) if itls else 0.0
    return tpot, max_itl
