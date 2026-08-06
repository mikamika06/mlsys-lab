import ref

def check(workdir):
    from prefill.simulate import simulate_batch
    cases = ref.get_simulation_cases()
    match_count = 0
    max_rel_err = 0.0
    for i, c in enumerate(cases):
        want = ref.simulate_batch(c)
        try:
            got = simulate_batch(c)
        except Exception as e:
            return {"sim_match": 0.0, "rel_err": 1.0, "_note": f"case {i} raised {e}"}
        if got == want:
            match_count += 1
        else:
            t_want = want["tokens_used"]
            t_got = got.get("tokens_used", 0)
            err = abs(t_got - t_want) / (t_want + 1e-9)
            if err > max_rel_err:
                max_rel_err = err
    sim_match = 1.0 if match_count == len(cases) else 0.0
    return {"sim_match": sim_match, "rel_err": float(max_rel_err)}
