import numpy as np

PREFILL_THROUGHPUT = 8000.0   # tokens/sec, single prefill engine
DECODE_RATE = 40.0            # tokens/sec/request, baseline (uninterrupted) decode speed
INTERFERENCE_FACTOR = 3.0     # colocated ITL inflation while a prefill overlaps the decode
TRANSFER_LATENCY = 0.04       # sec, KV-cache handoff overhead added to TTFT when disaggregated


def _make_trace(rng, n):
    intervals = rng.exponential(scale=0.15, size=n)
    arrival = np.cumsum(intervals)
    prompt_len = rng.integers(50, 1500, size=n).astype(np.float64)
    output_len = rng.integers(10, 150, size=n).astype(np.float64)
    ttft_slo = rng.choice([0.05, 0.15, 0.5], size=n, p=[0.3, 0.4, 0.3])
    itl_slo = rng.choice([0.03, 0.05, 0.12], size=n, p=[0.3, 0.4, 0.3])
    return arrival, prompt_len, output_len, ttft_slo, itl_slo


def _oracle(arrival, prompt_len, output_len, ttft_slo, itl_slo):
    prefill_dur = prompt_len / PREFILL_THROUGHPUT
    ttft_time = arrival + prefill_dur                    # absolute time of first token
    decode_end = ttft_time + output_len / DECODE_RATE    # absolute end of decode (baseline duration)

    prefill_start = arrival
    prefill_end = arrival + prefill_dur

    # request i's decode window [ttft_time_i, decode_end_i] overlaps request
    # j's prefill window [prefill_start_j, prefill_end_j], j != i
    overlap = (prefill_start[None, :] < decode_end[:, None]) & (prefill_end[None, :] > ttft_time[:, None])
    np.fill_diagonal(overlap, False)
    interferes = overlap.any(axis=1)

    itl_base = 1.0 / DECODE_RATE

    ttft_latency_colo = prefill_dur
    itl_colo = np.where(interferes, itl_base * INTERFERENCE_FACTOR, itl_base)
    meets_colo = (ttft_latency_colo <= ttft_slo) & (itl_colo <= itl_slo)

    ttft_latency_dis = prefill_dur + TRANSFER_LATENCY
    itl_dis = np.full(prompt_len.shape[0], itl_base)
    meets_dis = (ttft_latency_dis <= ttft_slo) & (itl_dis <= itl_slo)

    return int(meets_colo.sum()), int(meets_dis.sum())


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    arrival, prompt_len, output_len, ttft_slo, itl_slo = _make_trace(rng, 30)

    ref = _oracle(arrival, prompt_len, output_len, ttft_slo, itl_slo)

    try:
        got = sol.goodput_colocated_vs_disaggregated(
            arrival.copy(), prompt_len.copy(), output_len.copy(), ttft_slo.copy(), itl_slo.copy()
        )
        got_colo, got_dis = int(got[0]), int(got[1])
    except Exception:
        return {"exact_match": 0.0}

    ok = 1.0 if (got_colo, got_dis) == ref else 0.0
    return {"exact_match": ok}
