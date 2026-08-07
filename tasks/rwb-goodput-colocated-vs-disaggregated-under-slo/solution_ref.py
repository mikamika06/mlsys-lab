PREFILL_THROUGHPUT = 8000.0
DECODE_RATE = 40.0
INTERFERENCE_FACTOR = 3.0
TRANSFER_LATENCY = 0.04


def goodput_colocated_vs_disaggregated(arrival: list[float], prompt_len: list[float], output_len: list[float], ttft_slo: list[float], itl_slo: list[float]) -> tuple[int, int]:
    """
    Compute goodput for colocated vs disaggregated architectures.
    """
    n = len(prompt_len)

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
