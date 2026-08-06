"""Benchmark decode throughput with and without token mask constraints."""

import time
from typing import Any, Callable, Dict, List, Set, Tuple


def measure_throughput(
    vocab: Dict[int, str],
    eos_token_id: int,
    schema: Dict[str, Any],
    logits_fn: Callable[[List[int]], List[float]],
    max_tokens: int = 50,
) -> Dict[str, float]:
    raise NotImplementedError
