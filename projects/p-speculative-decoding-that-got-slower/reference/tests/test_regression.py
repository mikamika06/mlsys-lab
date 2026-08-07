import sys
sys.path.insert(0, ".")
from specdec.policy import AdaptivePolicy, evaluate_policy

def cost_model(b, gamma):
    tt = 10.0 + 1.0 * b
    td = 3.0 + 0.5 * b
    tv = 10.0 + 1.2 * b + 0.5 * b * gamma
    return td, tt, tv

def test_p95_never_degrades():
    gamma = 4
    reqs = [{"id": i, "b": 32, "p_true": 0.5, "domain": "mix"} for i in range(100)]

    base_p95 = evaluate_policy(reqs, lambda d, b: False, cost_model, gamma)

    pol = AdaptivePolicy(cost_model, gamma, default_p=0.5)
    for r in reqs:
        pol.update(r["domain"], gamma, int(gamma * r["p_true"]))

    spec_p95 = evaluate_policy(reqs, pol.decide, cost_model, gamma)

    assert spec_p95 <= base_p95 + 1e-5, f"P95 degraded: {spec_p95} > {base_p95}"

def test_speedup_on_good_traffic():
    gamma = 4
    reqs = [{"id": i, "b": 2, "p_true": 0.9, "domain": "easy"} for i in range(100)]

    base_mean = sum([cost_model(r["b"], gamma)[1] for r in reqs]) / len(reqs)

    pol = AdaptivePolicy(cost_model, gamma, default_p=0.9)
    for r in reqs:
        pol.update(r["domain"], gamma, int(gamma * r["p_true"]))

    tpts = []
    for r in reqs:
        if pol.decide(r["domain"], r["b"]):
            td, tt, tv = cost_model(r["b"], gamma)
            e_toks = 1.0 + (0.9 - 0.9**5) / 0.1
            tpts.append((gamma * td + tv) / e_toks)
        else:
            tpts.append(cost_model(r["b"], gamma)[1])

    spec_mean = sum(tpts) / len(tpts)
    assert spec_mean < base_mean * 0.9, "No significant speedup on good traffic"
