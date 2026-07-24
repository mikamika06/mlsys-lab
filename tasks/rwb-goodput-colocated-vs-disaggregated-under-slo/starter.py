import numpy as np

PREFILL_THROUGHPUT = 8000.0
DECODE_RATE = 40.0
INTERFERENCE_FACTOR = 3.0
TRANSFER_LATENCY = 0.04


def goodput_colocated_vs_disaggregated(arrival, prompt_len, output_len, ttft_slo, itl_slo):
    """
    Return (goodput_colocated, goodput_disaggregated): the count of requests
    that meet BOTH their TTFT and ITL SLO under each serving architecture.
    See task.md for the exact interference / transfer-latency model and the
    fixed PREFILL_THROUGHPUT / DECODE_RATE / INTERFERENCE_FACTOR /
    TRANSFER_LATENCY constants above.
    """
    raise NotImplementedError('your code here')
