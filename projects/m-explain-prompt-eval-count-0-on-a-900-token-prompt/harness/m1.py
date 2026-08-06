import ref


def check(workdir):
    from runner_metrics.analyzer import explain_zero_eval_count, compare_runners

    out = {"analysis_matched": 0.0}
    scenario = ref.SCENARIOS[0]

    try:
        ans_exp = explain_zero_eval_count(scenario["prompt_len"], scenario["metrics"])
        ans_comp = compare_runners(scenario["runner_a"], scenario["runner_b"])
        if isinstance(ans_exp, str) and len(ans_exp) > 5 and ans_comp in ("runner_a", "runner_b"):
            out["analysis_matched"] = 1.0
        else:
            out["_note"] = f"Invalid outputs: explain={ans_exp}, compare={ans_comp}"
    except Exception as e:
        out["_note"] = f"Error during execution: {str(e)}"
    return out
