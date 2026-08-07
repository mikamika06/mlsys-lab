def goodput_colocated_vs_disaggregated(arrival: list[float], prompt_len: list[float], output_len: list[float], ttft_slo: list[float], itl_slo: list[float]) -> tuple[int, int]:
    """
    Return (goodput_colocated, goodput_disaggregated): the count of requests
    that meet BOTH their TTFT and ITL SLO under each serving architecture.
    See task.md for the exact interference / transfer-latency model and the
    fixed PREFILL_THROUGHPUT / DECODE_RATE / INTERFERENCE_FACTOR /
    TRANSFER_LATENCY constants above.
    """
    raise NotImplementedError('your code here')
