import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from aotbreak.breakeven import compute_breakeven
    from aotbreak.schedule import select_strategy

    records = ref.generate_records()
    profiles = ref.parse_overhead_records(records)
    exp_calls = ref.EXPECTED_CALLS

    want_be = {w: ref.compute_breakeven(p) for w, p in profiles.items()}
    want_sched = ref.select_strategy(profiles, exp_calls)

    out = {"counts_matched": 0.0, "strategy_matched": 0.0, "rel_err": 1.0}

    max_rel_err = 0.0
    counts_ok = True
    for w, p in profiles.items():
        got_be = compute_breakeven(p)
        w_be = want_be[w]

        if got_be.get("preferred_mode") != w_be["preferred_mode"]:
            counts_ok = False
            out["_note"] = f"Preferred mode mismatch for {w}: got {got_be.get('preferred_mode')}, want {w_be['preferred_mode']}"

        for num_k in ["break_even_calls", "crossover_latency_ms", "overhead_delta_ms"]:
            w_val = w_be[num_k]
            g_val = got_be.get(num_k, 0.0)
            err = abs(g_val - w_val) / max(1e-9, abs(w_val))
            if err > max_rel_err:
                max_rel_err = err
            if err > 1e-5:
                counts_ok = False

    got_sched = select_strategy(profiles, exp_calls)
    sched_ok = True
    for w in profiles:
        w_sc = want_sched[w]
        g_sc = got_sched.get(w, {})

        if g_sc.get("selected_mode") != w_sc["selected_mode"]:
            sched_ok = False
            out["_note"] = f"Strategy mode mismatch for {w}: got {g_sc.get('selected_mode')}, want {w_sc['selected_mode']}"

        for num_k in ["estimated_latency_ms", "savings_ms"]:
            w_val = w_sc[num_k]
            g_val = g_sc.get(num_k, 0.0)
            err = abs(g_val - w_val) / max(1e-9, abs(w_val))
            if err > max_rel_err:
                max_rel_err = err
            if err > 1e-5:
                sched_ok = False

    out["rel_err"] = float(max_rel_err)
    out["counts_matched"] = 1.0 if counts_ok else 0.0
    out["strategy_matched"] = 1.0 if sched_ok else 0.0
    return out
