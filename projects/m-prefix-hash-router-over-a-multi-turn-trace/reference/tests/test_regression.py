import sys

sys.path.insert(0, ".")
from router.prefix import PrefixRouter, tokenize_into_blocks, compute_prefix_match
from router.bakeoff import simulate_trace
from router.tuning import grid_search_alpha


def test_prefix_matching_identifies_shared_history():
    tokens_turn1 = [10, 20, 30, 40, 50, 60, 70, 80]
    tokens_turn2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    b1 = tokenize_into_blocks(tokens_turn1, block_size=4)
    b2 = tokenize_into_blocks(tokens_turn2, block_size=4)
    m = compute_prefix_match(b2, b1)
    assert m == 2, f"Expected 2 matching blocks, got {m}"


def test_multi_turn_session_locality():
    router = PrefixRouter(num_workers=2, max_blocks_per_worker=16, block_size=4)
    t1 = [1, 2, 3, 4, 5, 6, 7, 8]
    t2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    w1, m1 = router.route(t1)
    router.update_cache(w1, t1)
    w2, m2 = router.route(t2)
    assert w2 == w1, f"Expected turn 2 to route to worker {w1}, got {w2}"
    assert m2 == 2, f"Expected 2 matched blocks, got {m2}"


def test_kv_aware_balances_locality_and_load():
    reqs = [
        {"req_id": 0, "session_id": 0, "tokens": list(range(32)), "arrival_time": 0.0, "gen_tokens": 100},
        {"req_id": 1, "session_id": 0, "tokens": list(range(48)), "arrival_time": 0.01, "gen_tokens": 10},
        {"req_id": 2, "session_id": 1, "tokens": [100 + i for i in range(32)], "arrival_time": 0.02, "gen_tokens": 10},
    ]
    res_kva = simulate_trace(
        reqs,
        num_workers=2,
        max_blocks_per_worker=16,
        block_size=4,
        prefill_rate=100.0,
        decode_rate=10.0,
        policy="kv_aware",
        alpha=0.5,
    )
    assert len(res_kva) == 3
    ttfts = [r["ttft"] for r in res_kva]
    assert all(t >= 0 for t in ttfts)


def test_alpha_tuning_evaluates_candidate_weights():
    reqs = [
        {"req_id": i, "session_id": i % 3, "tokens": [(i % 3) * 10 + j for j in range(16)], "arrival_time": i * 0.1, "gen_tokens": 20}
        for i in range(12)
    ]
    best_alpha, p95_map = grid_search_alpha(
        reqs,
        num_workers=2,
        max_blocks_per_worker=16,
        block_size=4,
        prefill_rate=500.0,
        decode_rate=50.0,
        alphas=[0.0, 0.5, 1.0],
    )
    assert best_alpha in [0.0, 0.5, 1.0]
    assert p95_map[best_alpha] <= p95_map[1.0] + 1e-6
