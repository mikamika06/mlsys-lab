from typing import Any, Dict, List


class ChunkedPrefillScheduler:
    """Schedules prefill and decode tokens while managing KV cache block health."""

    def __init__(self, token_budget: int = 512, cleanup_interval: int = 10) -> None:
        self.token_budget = token_budget
        self.cleanup_interval = cleanup_interval
        self.step_count = 0

    def step(self, active_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.step_count += 1
        processed_tokens = 0
        total_frag_blocks = 0

        for req in active_requests:
            rem = req.get("rem_prefill", 0)
            if rem > 0:
                take = min(rem, self.token_budget - processed_tokens)
                req["rem_prefill"] = rem - take
                processed_tokens += take
            else:
                processed_tokens += 1
            total_frag_blocks += req.get("frag_blocks", 0)
            if processed_tokens >= self.token_budget:
                break

        compute_latency = float(processed_tokens * 0.1)
        cleanup_latency = float(total_frag_blocks * 0.02)
        total_latency = compute_latency + cleanup_latency

        return {
            "step": self.step_count,
            "processed_tokens": processed_tokens,
            "compute_latency": compute_latency,
            "cleanup_latency": cleanup_latency,
            "total_latency": total_latency,
        }
