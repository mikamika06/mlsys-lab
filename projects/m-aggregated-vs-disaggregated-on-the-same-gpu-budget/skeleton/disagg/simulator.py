class Request:
    """Represents an LLM serving request."""

    def __init__(self, req_id: int, arrival_time: float, prompt_len: int, decode_len: int):
        self.req_id = req_id
        self.arrival_time = arrival_time
        self.prompt_len = prompt_len
        self.decode_len = decode_len


def simulate_aggregated(requests: list[Request], num_gpus: int, prefill_rate: float, decode_rate: float) -> list[dict]:
    """
    Simulates aggregated serving on N GPUs.
    Requests are load balanced round-robin across GPUs.
    Each GPU processes prefill sequentially, then decode tokens step-by-step.
    Returns list of dicts with: req_id, ttft, avg_itl, end_time.
    """
    raise NotImplementedError


def simulate_disaggregated(
    requests: list[Request],
    num_prefill_gpus: int,
    num_decode_gpus: int,
    prefill_rate: float,
    decode_rate: float,
    kv_transfer_rate: float,
    bytes_per_token: int = 1024,
) -> list[dict]:
    """
    Simulates disaggregated serving across P prefill GPUs and D decode GPUs.
    Prefill GPUs execute prefill, then transfer KV cache to decode GPUs.
    Transfer time = (prompt_len * bytes_per_token) / kv_transfer_rate.
    Decode GPUs perform token generation steps.
    Returns list of dicts with: req_id, ttft, avg_itl, end_time.
    """
    raise NotImplementedError
