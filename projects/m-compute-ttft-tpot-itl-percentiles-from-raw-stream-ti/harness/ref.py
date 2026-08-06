import numpy as np

def compute_metrics(raw_streams):
    ttfts = []
    tpots = []
    itls = []
    for stream in raw_streams:
        req_id, prompt_len, t_sent, timestamps = stream
        if not timestamps:
            continue
        ttft = timestamps[0] - t_sent
        ttfts.append(ttft)
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            itls.extend(list(intervals))
            total_gen_time = timestamps[-1] - timestamps[0]
            tpot = total_gen_time / (len(timestamps) - 1)
            tpots.append(tpot)
        else:
            tpots.append(0.0)
    return {
        "ttft_p50": float(np.percentile(ttfts, 50)) if ttfts else 0.0,
        "ttft_p99": float(np.percentile(ttfts, 99)) if ttfts else 0.0,
        "tpot_p50": float(np.percentile(tpots, 50)) if tpots else 0.0,
        "tpot_p99": float(np.percentile(tpots, 99)) if tpots else 0.0,
        "itl_p50": float(np.percentile(itls, 50)) if itls else 0.0,
        "itl_p99": float(np.percentile(itls, 99)) if itls else 0.0,
    }

def generate_trace(num_requests, total_tokens, prefix_ratio, seed=42):
    rng = np.random.default_rng(seed)
    shared_len = int(total_tokens * prefix_ratio)
    shared_prefix = list(rng.integers(1, 1000, size=shared_len))
    trace = []
    pool = list(rng.integers(1000, 10000, size=total_tokens * 3))
    idx = 0
    for _ in range(num_requests):
        req_len = int(rng.integers(50, 150))
        p_len = int(req_len * prefix_ratio)
        u_len = req_len - p_len
        req_tokens = shared_prefix[:p_len] + list(pool[idx:idx+u_len])
        idx += u_len
        trace.append(req_tokens)
    return trace

def validate_run(run_data):
    for req in run_data:
        if len(req.get("timestamps", [])) < 2:
            return False
        timestamps = req["timestamps"]
        if not timestamps:
            return False
        if any(timestamps[i] > timestamps[i+1] for i in range(len(timestamps)-1)):
            return False
        if req.get("stalled", False):
            return False
    return True
