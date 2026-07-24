import numpy as np


def _chunked_max(prompt_lengths, decode_load, budget):
    """FIFO continuous-batching scheduler: each iteration, decode_load[t]
    tokens are processed unconditionally (already-running decode
    requests can't be delayed); whatever budget remains is spent on
    prefill tokens taken off the front of the queue, packing as many
    (partial or whole) jobs as fit. Returns the max prefill tokens
    processed in any single iteration."""
    queue = [int(p) for p in prompt_lengths]
    t = 0
    max_step = 0
    guard = 0
    while queue:
        guard += 1
        if guard > 2_000_000:
            raise RuntimeError("chunked simulation did not terminate")
        d = decode_load[t] if t < len(decode_load) else 0
        remaining = max(budget - d, 0)
        step_prefill = 0
        while queue and remaining > 0:
            take = min(queue[0], remaining)
            queue[0] -= take
            remaining -= take
            step_prefill += take
            if queue[0] == 0:
                queue.pop(0)
        max_step = max(max_step, step_prefill)
        t += 1
    return max_step


def _unchunked_max(prompt_lengths):
    """Without chunking, each queued prompt is prefilled to completion in
    a single iteration (no cap). The worst single iteration is therefore
    exactly the largest prompt in the workload."""
    return max((int(p) for p in prompt_lengths), default=0)


def _oracle(prompt_lengths, decode_load, budget):
    return (
        _chunked_max(prompt_lengths, decode_load, budget),
        _unchunked_max(prompt_lengths),
    )


def _hand_cases():
    cases = []
    cases.append(([500, 20, 700], [0] * 10, 128))
    cases.append(([300], [50] * 20, 128))
    cases.append(([], [0, 0, 0], 128))
    cases.append(([10], [0], 128))
    cases.append(([40, 40, 40, 40], [0] * 10, 100))
    cases.append(([1000], [0] * 50, 1))
    return cases


def _gen_case(rng):
    n_jobs = int(rng.integers(1, 6))
    prompt_lengths = list(rng.integers(1, 900, size=n_jobs))
    n_iters = int(rng.integers(5, 20))
    budget = int(rng.integers(16, 300))
    decode_load = list(rng.integers(0, budget, size=n_iters))
    return prompt_lengths, decode_load, budget


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = _hand_cases()
    for _ in range(10):
        cases.append(_gen_case(rng))

    exact = 1.0
    for prompt_lengths, decode_load, budget in cases:
        ref = _oracle(prompt_lengths, decode_load, budget)
        try:
            got = sol.prefill_chunking_max_step(
                list(prompt_lengths), list(decode_load), budget
            )
            got = tuple(int(x) for x in got)
        except Exception:
            exact = 0.0
            break
        if got != ref:
            exact = 0.0
            break

    return {"exact_match": exact}
