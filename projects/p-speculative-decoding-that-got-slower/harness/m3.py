def check(workdir):
    from specdec.model import SpeculativeModel

    out = {
        "batched_overhead_correct": 0.0,
        "crossover_batch_size_matched": 0.0
    }

    model = SpeculativeModel(target_step_cost=10.0, draft_step_cost=1.0, overhead_per_draft=0.1)

    cost_b1 = model.expected_step_cost(gamma=4, batch_size=1)
    cost_b16 = model.expected_step_cost(gamma=4, batch_size=16)

    bf = 1.0 + 0.05 * 15
    draft_c = 4 * (1.0 * bf + 0.1)
    target_c = 10.0 * (1.0 + 0.02 * 4) * bf
    ref_cost_b16 = draft_c + target_c

    if abs(cost_b16 - ref_cost_b16) < 1e-5 and cost_b16 > cost_b1:
        out["batched_overhead_correct"] = 1.0

    crossover = model.batched_crossover_point(gamma=4, tau=0.6, max_batch=64)

    s_before = model.expected_speedup(gamma=4, tau=0.6, batch_size=crossover - 1)
    s_at = model.expected_speedup(gamma=4, tau=0.6, batch_size=crossover)

    if s_before >= 1.0 and s_at < 1.0:
        out["crossover_batch_size_matched"] = 1.0

    return out
