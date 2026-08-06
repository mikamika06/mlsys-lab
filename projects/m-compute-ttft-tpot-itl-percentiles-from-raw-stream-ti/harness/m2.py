import ref

def check(workdir):
    from streamstat.trace import generate_trace
    num_reqs = 20
    total_tokens = 2000
    ratio = 0.6
    seed = 123
    got_trace = generate_trace(num_reqs, total_tokens, ratio, seed=seed)
    want_trace = ref.generate_trace(num_reqs, total_tokens, ratio, seed=seed)
    match = 1.0 if got_trace == want_trace else 0.0
    return {"ratio_match": match}
