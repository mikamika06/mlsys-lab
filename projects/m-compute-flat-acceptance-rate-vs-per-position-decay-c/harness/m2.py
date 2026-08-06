import numpy as np
import ref


def check(workdir):
    out = {"domain_rate_rel_err": 1.0, "kl_bound_rel_err": 1.0}
    try:
        from spec.domains import analyze_domain_acceptance
        from spec.kl_bounds import kl_divergence_to_acceptance_bound
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    curves = ref.generate_domain_curves()
    draft_length = 4
    want_domains = ref.analyze_domain_acceptance(curves, draft_length)

    try:
        got_domains = analyze_domain_acceptance(curves, draft_length)
    except Exception as e:
        out["_note"] = f"Domain analysis error: {e}"
        return out

    domain_errs = []
    for k in want_domains:
        if k not in got_domains:
            out["_note"] = f"Missing domain key: {k}"
            return out
        err = abs(got_domains[k] - want_domains[k]) / (abs(want_domains[k]) + 1e-12)
        domain_errs.append(err)
    max_domain_err = max(domain_errs) if domain_errs else 1.0

    p_draft, p_target = ref.generate_prob_distributions()
    want_kl, want_bound = ref.kl_divergence_to_acceptance_bound(p_draft, p_target)

    try:
        got_kl, got_bound = kl_divergence_to_acceptance_bound(p_draft, p_target)
    except Exception as e:
        out["_note"] = f"KL bound calculation error: {e}"
        return out

    kl_err = abs(got_kl - want_kl) / (abs(want_kl) + 1e-12)
    bound_err = abs(got_bound - want_bound) / (abs(want_bound) + 1e-12)
    max_kl_err = max(kl_err, bound_err)

    out["domain_rate_rel_err"] = float(max_domain_err)
    out["kl_bound_rel_err"] = float(max_kl_err)
    return out
