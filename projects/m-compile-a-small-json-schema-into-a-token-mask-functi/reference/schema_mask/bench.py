import time
import numpy as np


def benchmark_decode(model_fn, masker, vocab: list[str], prompt_tokens: list[int], max_tokens: int = 20) -> dict:
    """Measures decoding performance with and without schema constraint."""
    tokens_unconstrained = list(prompt_tokens)
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        logits = model_fn(tokens_unconstrained)
        next_tok = int(np.argmax(logits))
        tokens_unconstrained.append(next_tok)
        if vocab[next_tok] == "<eos>":
            break
    t_unconstrained = time.perf_counter() - t0
    tok_s_unconstrained = len(tokens_unconstrained) / max(t_unconstrained, 1e-6)

    tokens_constrained = list(prompt_tokens)
    state = 0
    mask_time = 0.0
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        logits = model_fn(tokens_constrained)
        t_mask_start = time.perf_counter()
        mask = masker.get_mask(state)
        mask_time += time.perf_counter() - t_mask_start
        if not np.any(mask):
            break
        masked_logits = np.where(mask, logits, -1e9)
        next_tok = int(np.argmax(masked_logits))
        tokens_constrained.append(next_tok)
        state = masker.next_state(state, next_tok)
        if masker.is_terminal(state) or vocab[next_tok] == "<eos>":
            break
    t_constrained = time.perf_counter() - t0
    tok_s_constrained = len(tokens_constrained) / max(t_constrained, 1e-6)

    overhead_ms = (mask_time / max(len(tokens_constrained), 1)) * 1000.0

    return {
        "tok_s_unconstrained": float(tok_s_unconstrained),
        "tok_s_constrained": float(tok_s_constrained),
        "overhead_ms_per_tok": float(overhead_ms),
        "tokens_generated": int(len(tokens_constrained) - len(prompt_tokens)),
        "constrained_tokens": tokens_constrained,
    }
