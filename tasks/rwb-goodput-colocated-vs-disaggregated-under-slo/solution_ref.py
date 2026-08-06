import numpy as np

PREFILL_THROUGHPUT = 8000.0
DECODE_RATE = 40.0
INTERFERENCE_FACTOR = 3.0
TRANSFER_LATENCY = 0.04


def goodput_colocated_vs_disaggregated(arrival, prompt_len, output_len, ttft_slo, itl_slo):
    """
    Compute goodput for colocated vs disaggregated architectures.
    """
    arrival = np.asarray(arrival, dtype=np.float64)
    prompt_len = np.asarray(prompt_len, dtype=np.float64)
    output_len = np.asarray(output_len, dtype=np.float64)
    ttft_slo = np.asarray(ttft_slo, dtype=np.float64)
    itl_slo = np.asarray(itl_slo, dtype=np.float64)

    n = prompt_len.shape[0]

    prefill_dur = [0.0] * n
    ttft_time = [0.0] * n
    decode_end = [0.0] * n
    prefill_start = [0.0] * n
    prefill_end = [0.0] * n

    for i in range(n):
        p_dur = prompt_len[i] / PREFILL_THROUGHPUT
        arr = arrival[i]
        t_time = arr + p_dur
        d_end = t_time + output_len[i] / DECODE_RATE

        prefill_dur[i] = p_dur
        ttft_time[i] = t_time
        decode_end[i] = d_end
        prefill_start[i] = arr
        prefill_end[i] = arr + p_dur

    interferes = [False] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                if (prefill_start[j] < decode_end[i]) and (prefill_end[j] > ttft_time[i]):
                    interferes[i] = True
                    break

    itl_base = 1.0 / DECODE_RATE

    count_colo = 0
    count_dis = 0

    for i in range(n):
        ttft_latency_colo = prefill_dur[i]
        if interferes[i]:
            itl_colo = itl_base * INTERFERENCE_FACTOR
        else:
            itl_colo = itl_base

        if (ttft_latency_colo <= ttft_slo[i]) and (itl_colo <= itl_slo[i]):
            count_colo += 1

        ttft_latency_dis = prefill_dur[i] + TRANSFER_LATENCY
        itl_dis = itl_base

        if (ttft_latency_dis <= ttft_slo[i]) and (itl_dis <= itl_slo[i]):
            count_dis += 1

    return count_colo, count_dis
