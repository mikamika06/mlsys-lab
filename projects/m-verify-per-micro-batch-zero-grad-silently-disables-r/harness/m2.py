import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"bug_detected": 0.0, "effective_batch_matched": 0.0}

    try:
        from gradaccum.accumulator import SimpleModel, run_buggy_accumulation, run_correct_accumulation
        from gradaccum.tracer import analyze_accumulation_discrepancy
    except Exception as e:
        out["_note"] = f"Failed to import modules: {e}"
        return out

    weights = ref.make_weights(seed=42)
    mb_grads = ref.make_micro_batch_grads(num_mbs=12, seed=200)
    accum_steps = 4
    lr = 0.01

    try:
        model_corr = SimpleModel(weights)
        corr_grads = run_correct_accumulation(model_corr, mb_grads, accum_steps, lr)

        model_bug = SimpleModel(weights)
        bug_grads = run_buggy_accumulation(model_bug, mb_grads, accum_steps, lr)

        analysis = analyze_accumulation_discrepancy(corr_grads, bug_grads, accum_steps)
    except Exception as e:
        out["_note"] = f"Execution failed: {e}"
        return out

    want_analysis = ref.analyze_accumulation_discrepancy(corr_grads, bug_grads, accum_steps)

    if analysis.get("is_buggy") == 1.0 and analysis.get("max_abs_error", 0.0) > 1e-3:
        out["bug_detected"] = 1.0

    if ref.np.isclose(analysis.get("effective_batch_fraction", 0.0), want_analysis["effective_batch_fraction"]):
        out["effective_batch_matched"] = 1.0

    return out
