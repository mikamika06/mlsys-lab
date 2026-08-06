from typing import Callable, List, Any

def safe_generate_loop(
    model_step: Callable[[List[int]], int],
    prompt: List[int],
    max_new_tokens: int,
    cadence: int,
    empty_cache_fn: Callable[[], None]
) -> List[int]:
    tokens = list(prompt)
    for step in range(1, max_new_tokens + 1):
        next_tok = model_step(tokens)
        tokens.append(next_tok)
        if cadence > 0 and step % cadence == 0:
            empty_cache_fn()
    return tokens
