import heapq


class Request:
    """Represents an LLM serving request."""

    def __init__(self, req_id: int, arrival_time: float, prompt_len: int, decode_len: int):
        self.req_id = req_id
        self.arrival_time = arrival_time
        self.prompt_len = prompt_len
        self.decode_len = decode_len


def simulate_aggregated(requests: list[Request], num_gpus: int, prefill_rate: float, decode_rate: float) -> list[dict]:
    """Simulates aggregated serving on N GPUs with round-robin distribution."""
    if not requests:
        return []

    results = []
    gpu_available_time = [0.0] * num_gpus

    for i, req in enumerate(sorted(requests, key=lambda r: (r.arrival_time, r.req_id))):
        gpu_idx = i % num_gpus
        start_time = max(req.arrival_time, gpu_available_time[gpu_idx])

        prefill_dur = req.prompt_len / prefill_rate
        first_token_time = start_time + prefill_dur
        ttft = first_token_time - req.arrival_time

        decode_step_dur = 1.0 / decode_rate
        total_decode_dur = req.decode_len * decode_step_dur
        end_time = first_token_time + total_decode_dur

        avg_itl = decode_step_dur if req.decode_len > 0 else 0.0

        gpu_available_time[gpu_idx] = end_time
        results.append({
            "req_id": req.req_id,
            "ttft": ttft,
            "avg_itl": avg_itl,
            "end_time": end_time,
            "total_latency": end_time - req.arrival_time,
        })

    return sorted(results, key=lambda r: r["req_id"])


def simulate_disaggregated(
    requests: list[Request],
    num_prefill_gpus: int,
    num_decode_gpus: int,
    prefill_rate: float,
    decode_rate: float,
    kv_transfer_rate: float,
    bytes_per_token: int = 1024,
) -> list[dict]:
    """Simulates disaggregated serving across P prefill GPUs and D decode GPUs."""
    if not requests:
        return []

    sorted_reqs = sorted(requests, key=lambda r: (r.arrival_time, r.req_id))
    results = []

    p_available = [0.0] * num_prefill_gpus
    d_available = [0.0] * num_decode_gpus

    for i, req in enumerate(sorted_reqs):
        p_idx = i % num_prefill_gpus
        d_idx = i % num_decode_gpus

        p_start = max(req.arrival_time, p_available[p_idx])
        prefill_dur = req.prompt_len / prefill_rate
        prefill_end = p_start + prefill_dur
        p_available[p_idx] = prefill_end

        transfer_bytes = req.prompt_len * bytes_per_token
        transfer_dur = transfer_bytes / kv_transfer_rate
        transfer_end = prefill_end + transfer_dur

        d_start = max(transfer_end, d_available[d_idx])
        first_token_time = d_start + (1.0 / decode_rate)
        ttft = first_token_time - req.arrival_time

        total_decode_dur = req.decode_len * (1.0 / decode_rate)
        end_time = d_start + total_decode_dur
        d_available[d_idx] = end_time

        avg_itl = (1.0 / decode_rate) if req.decode_len > 0 else 0.0

        results.append({
            "req_id": req.req_id,
            "ttft": ttft,
            "avg_itl": avg_itl,
            "end_time": end_time,
            "total_latency": end_time - req.arrival_time,
        })

    return sorted(results, key=lambda r: r["req_id"])
