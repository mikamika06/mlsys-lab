import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from specdec.metrics import expected_accepted_tokens, expected_total_tokens, measure_acceptance_rates

    out = {
        "acceptance_rel_err": 1.0,
        "expected_tokens_rel_err": 1.0,
    }

    max_acc_err = 0.0
    max_exp_err = 0.0

    for trace in ref.TEST_TRACES:
        counts = trace["accepted_counts"]
        d_max = trace["draft_max"]

        want_rates = ref.measure_acceptance_rates(counts, d_max)
        got_rates = measure_acceptance_rates(counts, d_max)

        for w, g in zip(want_rates, got_rates):
            err = abs(w - g) / max(1e-9, abs(w))
            if err > max_acc_err:
                max_acc_err = err

        want_exp = ref.expected_accepted_tokens(want_rates)
        got_exp = expected_accepted_tokens(got_rates)
        err = abs(want_exp - got_exp) / max(1e-9, abs(want_exp))
        if err > max_exp_err:
            max_exp_err = err

        want_tot = ref.expected_total_tokens(want_rates)
        got_tot = expected_total_tokens(got_rates)
        err = abs(want_tot - got_tot) / max(1e-9, abs(want_tot))
        if err > max_exp_err:
            max_exp_err = err

    out["acceptance_rel_err"] = float(max_acc_err)
    out["expected_tokens_rel_err"] = float(max_exp_err)
    return out
