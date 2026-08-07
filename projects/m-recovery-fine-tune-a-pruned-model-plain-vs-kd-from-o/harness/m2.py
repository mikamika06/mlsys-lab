import numpy as np
import ref


def check(workdir):
    from recovery.eval import evaluate_accuracy, steps_to_90_recovery

    X, y, teacher, student, accuracies, baseline, pruned = ref.generate_fixture()
    out = {"steps_matched": 0.0}
    try:
        acc_got = evaluate_accuracy(student, X, y)
        acc_want = ref.ref_evaluate_accuracy(student, X, y)
        steps_got = steps_to_90_recovery(accuracies, baseline, pruned)
        steps_want = ref.ref_steps_to_90_recovery(accuracies, baseline, pruned)
        if np.isclose(acc_got, acc_want, atol=1e-5) and steps_got == steps_want:
            out["steps_matched"] = 1.0
        else:
            out["_note"] = f"eval mismatch: acc got {acc_got}, want {acc_want}; steps got {steps_got}, want {steps_want}"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)[:100]}"
    return out
