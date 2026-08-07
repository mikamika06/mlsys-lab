import ref

def check(workdir):
    from medusa.sampling import simulate_speculative_sampling
    got_typ = simulate_speculative_sampling(None, None, "typical")
    got_str = simulate_speculative_sampling(None, None, "strict")
    want_typ = 1.85
    want_str = 1.42
    match = 1.0 if (got_typ > got_str and abs(got_typ - want_typ) < 0.3 and abs(got_str - want_str) < 0.3) else 0.0
    return {"sampling_match": float(match)}
