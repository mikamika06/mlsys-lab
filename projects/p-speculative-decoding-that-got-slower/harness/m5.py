import sys
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        import specdec.policy as pol
    except ImportError:
        return {"p95_ok": 0.0}

    m = {"p95_ok": 0.0}
    reqs = ref.generate_requests()

    try:
        p95_always = pol.evaluate_policy(reqs, lambda d, b: True, ref.cost_model, 4)
        p95_never = pol.evaluate_policy(reqs, lambda d, b: False, ref.cost_model, 4)

        tpts_always = []
        for r in reqs:
            td, tt, tv = ref.cost_model(r["b"], 4)
            p = r["p_true"]
            e = 1.0 + (p - p**5)/(1.0-p)
            tpts_always.append((4*td + tv)/e)
        tpts_always.sort()
        expected_always = tpts_always[int(0.95 * len(tpts_always))]

        tpts_base = sorted([ref.cost_model(r["b"], 4)[1] for r in reqs])
        expected_never = tpts_base[int(0.95 * len(tpts_base))]

        if abs(p95_always - expected_always) < 1e-4 and abs(p95_never - expected_never) < 1e-4:
            m["p95_ok"] = 1.0
    except Exception:
        pass

    return m
