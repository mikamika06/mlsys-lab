import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from specdec.tuning import evaluate_draft_throughput, select_optimal_draft_max

    out = {
        "optimal_gamma_matched": 0.0,
        "throughput_rel_err": 1.0,
    }

    matched_count = 0
    total_count = len(ref.TEST_TRACES)
    max_tp_err = 0.0

    for trace in ref.TEST_TRACES:
        counts = trace["accepted_counts"]
        d_max = trace["draft_max"]
        d_time = trace["draft_time"]
        t_time = trace["target_time"]

        rates = ref.measure_acceptance_rates(counts, d_max)

        want_tp = ref.evaluate_draft_throughput(rates, d_time, t_time)
        got_tp = evaluate_draft_throughput(rates, d_time, t_time)

        for gamma in want_tp:
            w = want_tp[gamma]
            g = got_tp.get(gamma, 0.0)
            err = abs(w - g) / max(1e-9, abs(w))
            if err > max_tp_err:
                max_tp_err = err

        want_gamma, want_rate = ref.select_optimal_draft_max(rates, d_time, t_time)
        got_gamma, got_rate = select_optimal_draft_max(rates, d_time, t_time)

        if got_gamma == want_gamma and abs(want_rate - got_rate) / max(1e-9, abs(want_rate)) < 1e-5:
            matched_count += 1

    out["optimal_gamma_matched"] = 1.0 if matched_count == total_count else 0.0
    out["throughput_rel_err"] = float(max_tp_err)
    return out
