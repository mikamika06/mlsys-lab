import sys

sys.path.insert(0, ".")
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from evalrec.contamination import detect_contaminated_baseline
        from evalrec.gate import evaluate_acceptance_gate
    except Exception as e:
        return {
            "contamination_checks_pass": 0.0,
            "gate_evaluations_pass": 0.0,
            "_note": f"Import error: {type(e).__name__}: {e}",
        }

    teacher_data, historical_ppl = ref.get_m2_contamination_cases()
    try:
        want_contam = ref.detect_contaminated_baseline(
            teacher_data, historical_ppl
        )
        got_contam = detect_contaminated_baseline(teacher_data, historical_ppl)
    except Exception as e:
        return {
            "contamination_checks_pass": 0.0,
            "gate_evaluations_pass": 0.0,
            "_note": f"Contamination check error: {type(e).__name__}: {e}",
        }

    contam_ok = 1.0 if got_contam == want_contam else 0.0

    cand_metrics, thresh_pass, thresh_fail = ref.get_m2_gate_cases()
    try:
        want_pass = ref.evaluate_acceptance_gate(cand_metrics, thresh_pass)
        got_pass = evaluate_acceptance_gate(cand_metrics, thresh_pass)

        want_fail = ref.evaluate_acceptance_gate(cand_metrics, thresh_fail)
        got_fail = evaluate_acceptance_gate(cand_metrics, thresh_fail)
    except Exception as e:
        return {
            "contamination_checks_pass": contam_ok,
            "gate_evaluations_pass": 0.0,
            "_note": f"Gate evaluation error: {type(e).__name__}: {e}",
        }

    gate_ok = (
        1.0 if (got_pass == want_pass and got_fail == want_fail) else 0.0
    )

    out = {
        "contamination_checks_pass": contam_ok,
        "gate_evaluations_pass": gate_ok,
    }
    if not contam_ok:
        out[
            "_note"
        ] = f"Contamination mismatch: got {got_contam}, expected {want_contam}"
    elif not gate_ok:
        out[
            "_note"
        ] = f"Gate mismatch: pass_got={got_pass}, fail_got={got_fail}"

    return out
