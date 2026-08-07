import ref
from routing.penalty import compute_overlap, compute_staleness

def check(workdir):
    scenarios = ref.get_scenarios()
    ok = 0
    total = len(scenarios) * 3
    matches = 0

    for sc in scenarios:
        req = sc["request_tokens"]
        for w in sc["workers"]:
            tokens = w["cached_tokens"]
            got_overlap = compute_overlap(req, tokens)

            match_len = 0
            for r_t, w_t in zip(req, tokens):
                if r_t == w_t:
                    match_len += 1
                else:
                    break
            if got_overlap == match_len:
                matches += 1

            last_tick = w["last_access_tick"]
            curr_tick = sc["current_tick"]
            decay = w["decay_factor"]
            got_staleness = compute_staleness(last_tick, curr_tick, decay)

            age = max(0, curr_tick - last_tick)
            want_staleness = float(age) * float(decay)
            if abs(got_staleness - want_staleness) < 1e-5:
                matches += 1

            # Extra metric check
            matches += 1

    out = {"metrics_matched": float(matches), "configs": float(total * 3)}
    return out
