from typing import Callable, List, Any

def safe_generate_loop(
    model_step: Callable[[List[int]], int],
    prompt: List[int],
    max_new_tokens: int,
    cadence: int,
    empty_cache_fn: Callable[[], None]
) -> List[int]:
    """Runs generation loop, executing empty_cache_fn every `cadence` tokens."""
    raise NotImplementedError
