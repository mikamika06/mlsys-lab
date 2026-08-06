import sys
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        from divergence.analyze import check_regression_gate, analyze_near_ties
    except ImportError:
        return {"gate_correct": 0.0, "ties_correct": 0.0}

    divs, k, max_f = ref.get_fixtures_m2_gate()
    try:
        got_gate = check_regression_gate(divs, k, max_f)
        want_gate = ref.check_regression_gate(divs, k, max_f)
        gate_ok = 1.0 if got_gate == want_gate else 0.0
    except Exception:
        gate_ok = 0.0

    logits, divs2, eps = ref.get_fixtures_m2_ties()
    try:
        got_ties = analyze_near_ties(logits, divs2, eps)
        want_ties = ref.analyze_near_ties(logits, divs2, eps)
        ties_ok = 1.0 if abs(got_ties - want_ties) < 1e-5 else 0.0
    except Exception:
        ties_ok = 0.0

    return {"gate_correct": gate_ok, "ties_correct": ties_ok}
