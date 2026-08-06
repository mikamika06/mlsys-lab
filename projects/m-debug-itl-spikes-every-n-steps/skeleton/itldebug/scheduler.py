from typing import Any, Dict, List


class ChunkedPrefillScheduler:
    """Schedules prefill and decode tokens while managing KV cache block health."""

    def __init__(self, token_budget: int = 512, cleanup_interval: int = 10) -> None:
        raise NotImplementedError

    def step(self, active_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError
