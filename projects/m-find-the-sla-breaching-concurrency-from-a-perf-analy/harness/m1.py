import ref

def check(workdir):
    from perfanalysis.sweep import find_max_concurrency
    out = {"concurrency_matched": 0.0}
    slas = [40.0, 70.0, 100.0]
    data = ref.generate_sweep_data()
    matched = 0
    for sla in slas:
        want = ref.find_max_concurrency(data, sla)
        got = find_max_concurrency(data, sla)
        if abs(got - want) < 1e-5:
            matched += 1
    out["concurrency_matched"] = float(matched)
    return out
