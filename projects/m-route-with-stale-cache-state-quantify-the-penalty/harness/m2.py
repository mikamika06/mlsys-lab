import ref
from routing.cost import routing_cost, select_best_worker

def check(workdir):
    scenarios = ref.get_scenarios()
    cost_ok = 0
    optimal_ok = 0
    total = len(scenarios)

    for sc in scenarios:
        req = sc["request_tokens"]
        workers = sc["workers"]
        t_cost = sc["transfer_cost"]
        c_cost = sc["compute_cost"]
        curr_tick = sc["current_tick"]

        # Test routing cost
        got_cost = routing_cost(10, 20, 0.1, t_cost, c_cost)
        missing_len = max(0, 20 - 10)
        base_cost = float(10) * float(t_cost) + float(missing_len) * float(c_cost)
        want_cost = base_cost + base_cost * 0.1
        if abs(got_cost - want_cost) < 1e-5:
            cost_ok += 1

        # Test select best worker
        got_best = select_best_worker(req, workers, t_cost, c_cost, curr_tick)

        best_wid = None
        min_cost = float("inf")
        for w in workers:
            wid = w["worker_id"]
            tokens = w["cached_tokens"]
            last_tick = w["last_access_tick"]
            decay = w.get("decay_factor", 0.01)

            match_len = 0
            for r_t, w_t in zip(req, tokens):
                if r_t == w_t:
                    match_len += 1
                else:
                    break
            age = max(0, curr_tick - last_tick)
            staleness = float(age) * float(decay)

            missing = max(0, len(req) - match_len)
            b_cost = float(match_len) * float(t_cost) + float(missing) * float(c_cost)
            cost = b_cost + b_cost * staleness

            if cost < min_cost:
                min_cost = cost
                best_wid = wid

        if got_best == best_wid:
            optimal_ok += 1

    out = {
        "cost_match": 1.0 if cost_ok == total else 0.0,
        "optimal_match": 1.0 if optimal_ok == total else 0.0
    }
    return out
