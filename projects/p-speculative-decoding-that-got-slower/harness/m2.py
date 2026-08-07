def check(workdir):
    from specdec.model import SpeculativeModel

    out = {
        "speedup_model_accuracy": 0.0,
        "optimal_gamma_correct": 0.0
    }

    model = SpeculativeModel(target_step_cost=10.0, draft_step_cost=1.0, overhead_per_draft=0.1)

    exp_tok = model.expected_accepted_tokens(gamma=4, tau=0.5)
    exp_tok_ref = (1.0 - (0.5 ** 5)) / (1.0 - 0.5)

    exp_cost = model.expected_step_cost(gamma=4, batch_size=1)
    exp_cost_ref = 4 * (1.0 + 0.1) + 10.0 * (1.0 + 0.02 * 4)

    speedup = model.expected_speedup(gamma=4, tau=0.5, batch_size=1)
    speedup_ref = 10.0 / (exp_cost_ref / exp_tok_ref)

    if (abs(exp_tok - exp_tok_ref) < 1e-5 and
            abs(exp_cost - exp_cost_ref) < 1e-5 and
            abs(speedup - speedup_ref) < 1e-5):
        out["speedup_model_accuracy"] = 1.0

    opt_g_high = model.optimal_gamma(tau=0.9, max_gamma=8, batch_size=1)
    opt_g_low = model.optimal_gamma(tau=0.1, max_gamma=8, batch_size=1)

    if opt_g_high > 0 and opt_g_low == 0:
        out["optimal_gamma_correct"] = 1.0

    return out
