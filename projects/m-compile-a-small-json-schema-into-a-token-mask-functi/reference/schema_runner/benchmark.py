"""Benchmark decode throughput with and without token mask constraints."""

import time
from typing import Any, Callable, Dict, List, Set, Tuple
from schema_runner.compiler import SchemaMaskCompiler


def measure_throughput(
    vocab: Dict[int, str],
    eos_token_id: int,
    schema: Dict[str, Any],
    logits_fn: Callable[[List[int]], List[float]],
    max_tokens: int = 50,
) -> Dict[str, float]:
    compiler = SchemaMaskCompiler(vocab, eos_token_id)
    mask_fn = compiler.compile(schema)

    def run_decode(use_mask: bool) -> Tuple[int, float]:
        tokens = []
        t0 = time.perf_counter()
        for _ in range(max_tokens):
            logits = logits_fn(tokens)
            if use_mask:
                allowed = mask_fn(tokens)
                if not allowed:
                    break
                best_tok = max(allowed, key=lambda t: logits[t])
            else:
                best_tok = max(range(len(logits)), key=lambda t: logits[t])
            tokens.append(best_tok)
            if best_tok == eos_token_id:
                break
        elapsed = time.perf_counter() - t0
        return len(tokens), max(elapsed, 1e-6)

    unconstrained_toks, unconstrained_time = run_decode(use_mask=False)
    constrained_toks, constrained_time = run_decode(use_mask=True)

    unconstrained_tps = unconstrained_toks / unconstrained_time
    constrained_tps = constrained_toks / constrained_time

    return {
        "unconstrained_tok_s": unconstrained_tps,
        "constrained_tok_s": constrained_tps,
        "speedup_ratio": constrained_tps / max(unconstrained_tps, 1e-6),
    }
