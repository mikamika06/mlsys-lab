import ref


def check(workdir):
    from gkdstep.drift import measure_sequence_drift

    out = {"drift_rel_err": 1.0}
    max_err = 0.0
    betas = [0.0, 0.25, 0.5, 0.8]
    temperatures = [0.8, 1.0, 1.5]

    for t_logits, s_logits in ref.TEST_CASES:
        for beta in betas:
            for temp in temperatures:
                want = ref.measure_sequence_drift(t_logits, s_logits, beta=beta, temperature=temp)
                try:
                    got = measure_sequence_drift(t_logits, s_logits, beta=beta, temperature=temp)
                except Exception as e:
                    out["_note"] = f"Error in measure_sequence_drift: {type(e).__name__}: {e}"
                    return out

                if not isinstance(got, dict) or "mean_drift" not in got or "step_drifts" not in got:
                    out["_note"] = "Return value missing required keys"
                    return out

                err1 = abs(want["mean_drift"] - got["mean_drift"]) / (abs(want["mean_drift"]) + 1e-12)
                err2 = abs(want["off_policy_baseline"] - got["off_policy_baseline"]) / (
                    abs(want["off_policy_baseline"]) + 1e-12
                )
                err3 = float(
                    ref.np.max(
                        ref.np.abs(want["step_drifts"] - got["step_drifts"])
                        / (ref.np.abs(want["step_drifts"]) + 1e-12)
                    )
                )

                err = max(err1, err2, err3)
                if err > max_err:
                    max_err = err

    out["drift_rel_err"] = float(max_err)
    return out
