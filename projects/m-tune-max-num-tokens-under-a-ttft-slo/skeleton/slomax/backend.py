def estimate_step_latency(num_tokens: int, num_seqs: int, backend_type: str, batching_mode: str) -> float:
    raise NotImplementedError


def compute_ttft(prompt_length: int, max_num_tokens: int, backend_type: str, batching_mode: str) -> float:
    raise NotImplementedError
