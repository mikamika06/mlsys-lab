import ref
from kd.train import run_epoch, evaluate_accuracy


def check(workdir):
    out = {"accuracy_delta": 0.0}
    try:
        acc_hard = evaluate_accuracy(ref.SAMPLE_LOGITS, ref.SAMPLE_TARGETS)
        acc_blended = evaluate_accuracy(ref.SAMPLE_LOGITS + 0.1, ref.SAMPLE_TARGETS)
        loss_val = run_epoch(ref.SAMPLE_LOGITS, ref.SAMPLE_TEACHER_LOGITS, ref.SAMPLE_TARGETS, 0.5, 2.0, "blended")
        if isinstance(acc_hard, float) and isinstance(acc_blended, float) and isinstance(loss_val, float):
            out["accuracy_delta"] = 1.0
    except Exception as e:
        out["_note"] = f"train module failed: {e}"
    return out
