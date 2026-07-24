import numpy as np

PREFILL_THROUGHPUT = 8000.0
DECODE_RATE = 40.0
INTERFERENCE_FACTOR = 3.0
TRANSFER_LATENCY = 0.04


def goodput_colocated_vs_disaggregated(arrival, prompt_len, output_len, ttft_slo, itl_slo):
    """
    arrival, prompt_len, output_len, ttft_slo, itl_slo: 1-D float arrays,
    one entry per request (arrival time in seconds, prompt/output token
    counts, and the per-request TTFT/ITL SLO thresholds in seconds).

    Serving-engine constants (module-level, fixed):
      PREFILL_THROUGHPUT  tokens/sec a single prefill engine sustains.
      DECODE_RATE          tokens/sec/request a decode step normally sustains.
      INTERFERENCE_FACTOR  ITL inflation factor while a prefill overlaps a
                            decode on the SAME (colocated) engine.
      TRANSFER_LATENCY     extra seconds added to TTFT when prefill and
                            decode run on separate (disaggregated) engines,
                            for the KV-cache handoff between them.

    For request i:
      prefill_dur_i = prompt_len_i / PREFILL_THROUGHPUT
      ttft_time_i   = arrival_i + prefill_dur_i          (absolute time of first token)
      decode_end_i  = ttft_time_i + output_len_i / DECODE_RATE

    Request i "interferes" if any OTHER request j's prefill window
    [arrival_j, arrival_j + prompt_len_j/PREFILL_THROUGHPUT] overlaps i's
    decode window [ttft_time_i, decode_end_i].

    COLOCATED:  ttft_latency_i = prefill_dur_i
                itl_i = (1/DECODE_RATE) * INTERFERENCE_FACTOR if interferes_i else (1/DECODE_RATE)
    DISAGGREGATED: ttft_latency_i = prefill_dur_i + TRANSFER_LATENCY
                   itl_i = 1/DECODE_RATE   (never inflated -- separate engines)

    A request meets SLO under an architecture iff BOTH its TTFT latency
    <= ttft_slo_i AND its ITL <= itl_slo_i under that architecture.

    Returns (goodput_colocated, goodput_disaggregated): plain Python ints,
    the COUNT of requests meeting both SLOs under each architecture.
    """
    arrival = np.asarray(arrival, dtype=np.float64)
    prompt_len = np.asarray(prompt_len, dtype=np.float64)
    output_len = np.asarray(output_len, dtype=np.float64)
    ttft_slo = np.asarray(ttft_slo, dtype=np.float64)
    itl_slo = np.asarray(itl_slo, dtype=np.float64)

    prefill_dur = prompt_len / PREFILL_THROUGHPUT
    ttft_time = arrival + prefill_dur
    decode_end = ttft_time + output_len / DECODE_RATE

    prefill_start = arrival
    prefill_end = arrival + prefill_dur

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
