"""Checker for Milestone 2: Break-even acceptance rate and prefill diagnostics."""

import os
import sys


def check(workdir):
    sys.path.insert(0, os.path.join(workdir, "harness"))
    sys.path.insert(0, workdir)
    import ref
    from specdec.efficiency import analyze_prefill_speculation, compute_breakeven_acceptance_rate

    be_cases = ref.generate_efficiency_cases()
    be_ok = True
    for c in be_cases:
        got = compute_breakeven_acceptance_rate(c["r"], c["gamma"])
        if abs(got - c["expected_alpha"]) > 1e-5:
            be_ok = False
            break

    pp_cases = ref.generate_prefill_cases()
    pp_ok = True
    for pc in pp_cases:
        diag = analyze_prefill_speculation(pc["length"], pc["gamma"], pc["draft_t"], pc["target_t"])
        if not isinstance(diag, dict):
            pp_ok = False
            break
        if diag.get("helps_prefill") is not False:
            pp_ok = False
            break
        if diag.get("speedup", 2.0) >= 1.0:
            pp_ok = False
            break

    return {
        "breakeven_matched": 1.0 if be_ok else 0.0,
        "prefill_measured": 1.0 if pp_ok else 0.0
    }
