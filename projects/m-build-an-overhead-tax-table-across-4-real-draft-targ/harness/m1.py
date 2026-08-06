import ref


def check(workdir):
    from tax.overhead import compute_pair_tax
    from tax.profiles import calculate_expected_acceptance

    out = {
        "acceptance_matched": 0.0,
        "overhead_matched": 0.0,
        "rel_err": 1.0,
    }

    max_err = 0.0
    acc_ok = True
    overhead_ok = True

    for pair in ref.PAIRS_DATA:
        for gamma in [2, 4, 5]:
            want_acc = ref.calculate_expected_acceptance(pair["acceptance_probs"], gamma)
            got_acc = calculate_expected_acceptance(pair["acceptance_probs"], gamma)

            for k in ["expected_accepted"]:
                w_val = float(want_acc[k])
                g_val = float(got_acc.get(k, 0.0))
                err = abs(g_val - w_val) / max(1.0, abs(w_val))
                max_err = max(max_err, err)
                if err > 1e-5:
                    acc_ok = False

            want_tax = ref.compute_pair_tax(pair, gamma)
            got_tax = compute_pair_tax(pair, gamma)

            for k in ["total_step_ms", "effective_latency_ms", "speedup", "overhead_tax"]:
                w_val = float(want_tax[k])
                g_val = float(got_tax.get(k, 0.0))
                err = abs(g_val - w_val) / max(1.0, abs(w_val))
                max_err = max(max_err, err)
                if err > 1e-5:
                    overhead_ok = False

    out["acceptance_matched"] = 1.0 if acc_ok else 0.0
    out["overhead_matched"] = 1.0 if overhead_ok else 0.0
    out["rel_err"] = float(max_err)
    return out
