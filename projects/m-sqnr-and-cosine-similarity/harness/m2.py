import numpy as np
import ref


def check(workdir):
    from numval.amplification import analyze_amplification
    from numval.gate import evaluate_gate

    out = {"gate_eval_matches": 0.0, "amplification_matches": 0.0}
    gate_ok = 0
    total_gate = len(ref.GATE_TEST_CASES)

    for i, (y_ref, y_test, min_sqnr, min_cos, max_rel) in enumerate(ref.GATE_TEST_CASES):
        want = ref.evaluate_gate(y_ref, y_test, min_sqnr, min_cos, max_rel)
        got = evaluate_gate(y_ref, y_test, min_sqnr_db=min_sqnr, min_cos_sim=min_cos, max_rel_err=max_rel)

        matches = (
            got.get("passed") == want["passed"]
            and np.isclose(got.get("sqnr_db", 0), want["sqnr_db"], rtol=1e-3, atol=1e-3)
            and np.isclose(got.get("cos_sim", 0), want["cos_sim"], rtol=1e-3, atol=1e-3)
            and np.isclose(got.get("max_rel_err", 0), want["max_rel_err"], rtol=1e-3, atol=1e-3)
        )
        if matches:
            gate_ok += 1
        elif "_note" not in out:
            out["_note"] = f"gate case {i}: got {got}, want {want}"

    amp_ok = 0
    total_amp = len(ref.LAYER_TEST_CASES)
    for i, (r_chain, t_chain) in enumerate(ref.LAYER_TEST_CASES):
        want = ref.analyze_amplification(r_chain, t_chain)
        got = analyze_amplification(r_chain, t_chain)

        err_match = np.allclose(got.get("layer_errors", []), want["layer_errors"], rtol=1e-3, atol=1e-3)
        amp_match = np.allclose(got.get("amplifications", []), want["amplifications"], rtol=1e-3, atol=1e-3)
        max_match = got.get("max_amplifying_layer") == want["max_amplifying_layer"]

        if err_match and amp_match and max_match:
            amp_ok += 1
        elif "_note" not in out:
            out["_note"] = f"amp case {i}: got {got}, want {want}"

    out["gate_eval_matches"] = float(gate_ok) / total_gate
    out["amplification_matches"] = float(amp_ok) / total_amp
    return out
