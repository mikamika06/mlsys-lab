import numpy as np

from router.prefix import (
    block_hash,
    tokenize_into_blocks,
    compute_prefix_match,
    PrefixRouter,
)
from router.bakeoff import simulate_trace, run_bakeoff, compute_p95_ttft
from router.tuning import grid_search_alpha


def generate_synthetic_trace(
    num_sessions: int = 10,
    turns_per_session: int = 4,
    tokens_per_turn: int = 64,
    seed: int = 42
) -> list[dict]:
    rng = np.random.default_rng(seed)
    requests = []
    req_id = 0

    session_histories = {s: [] for s in range(num_sessions)}
    session_arrival_times = {s: float(s * 0.5) for s in range(num_sessions)}

    for turn in range(turns_per_session):
        session_order = list(range(num_sessions))
        rng.shuffle(session_order)
        for s in session_order:
            turn_tokens = rng.integers(100, 5000, size=tokens_per_turn).tolist()
            session_histories[s].extend(turn_tokens)
            prompt_tokens = list(session_histories[s])
            gen_tokens = int(rng.integers(16, 64))

            arr_time = session_arrival_times[s] + float(rng.uniform(0.1, 0.5))
            session_arrival_times[s] = arr_time + 1.0

            requests.append({
                "req_id": req_id,
                "session_id": s,
                "tokens": prompt_tokens,
                "arrival_time": round(arr_time, 4),
                "gen_tokens": gen_tokens
            })
            req_id += 1

    requests.sort(key=lambda r: r["arrival_time"])
    return requests


TRACES = [
    generate_synthetic_trace(num_sessions=6, turns_per_session=3, seed=101),
    generate_synthetic_trace(num_sessions=12, turns_per_session=5, seed=202),
]
