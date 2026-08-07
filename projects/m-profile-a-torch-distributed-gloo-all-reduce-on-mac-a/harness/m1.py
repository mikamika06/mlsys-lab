import ref

def check(workdir):
    from glooprof.profiler import extract_gloo_self_time
    out = {"rel_err": 0.0}
    errors = []
    for t in ref.TRACES:
        want = t["expected_self_time"]
        got = extract_gloo_self_time(t)
        err = abs(got - want) / (abs(want) + 1e-9)
        errors.append(err)
    max_err = max(errors) if errors else 0.0
    out["rel_err"] = float(max_err)
    return out
