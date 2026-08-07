import numpy as np


def generate_arrivals(n, rate=0.05, seed=42):
    rng = np.random.RandomState(seed)
    inter_arrivals = rng.exponential(1.0 / rate, n)
    arrivals = np.cumsum(inter_arrivals).tolist()
    seq_lens = rng.randint(32, 512, n).tolist()
    return arrivals, seq_lens


def ref_profile_latency(batch_size, seq_len):
    return 10.0 + 0.5 * batch_size + 0.1 * seq_len + 0.05 * batch_size * seq_len


def ref_warmup_cost(batch_sizes, seq_lens):
    return sum(ref_profile_latency(b, s) for b in batch_sizes for s in seq_lens)


def ref_simulate(arrivals, seq_lens, max_batch, timeout):
    n = len(arrivals)
    latencies = []
    current_time = 0.0
    idx = 0
    queue = []

    while idx < n or queue:
        if not queue and idx < n and current_time < arrivals[idx]:
            current_time = arrivals[idx]

        while idx < n and arrivals[idx] <= current_time:
            queue.append((arrivals[idx], seq_lens[idx]))
            idx += 1

        if queue:
            wait_time = current_time - queue[0][0]
            if len(queue) >= max_batch or wait_time >= timeout - 1e-9:
                batch = queue[:max_batch]
                queue = queue[max_batch:]
                b_size = len(batch)
                b_seq = max(item[1] for item in batch)
                exec_time = ref_profile_latency(b_size, b_seq)

                finish_time = current_time + exec_time
                for arr, seq in batch:
                    latencies.append(finish_time - arr)

                current_time = finish_time
            else:
                next_timeout = queue[0][0] + timeout
                if idx < n:
                    current_time = min(arrivals[idx], next_timeout)
                else:
                    current_time = next_timeout

    s_lat = sorted(latencies)
    return {
        "p50": s_lat[int(len(s_lat) * 0.50)],
        "p99": s_lat[int(len(s_lat) * 0.99)]
    }


def ref_tune_batching(arrivals, seq_lens, target_p99):
    max_batches = [1, 2, 4, 8, 16]
    timeouts = [5.0, 10.0, 20.0, 50.0]

    best_config = None
    best_p50 = float('inf')
    fallback_config = None
    fallback_p99 = float('inf')

    for mb in max_batches:
        for to in timeouts:
            res = ref_simulate(arrivals, seq_lens, mb, to)
            if res["p99"] <= target_p99:
                if res["p50"] < best_p50:
                    best_p50 = res["p50"]
                    best_config = (mb, to)

            if res["p99"] < fallback_p99:
                fallback_p99 = res["p99"]
                fallback_config = (mb, to)

    return best_config if best_config else fallback_config
