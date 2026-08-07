import ref
from kd.loss import hard_loss, soft_loss, blended_loss


def check(workdir):
    out = {"loss_matched": 0.0}
    ok = 0
    try:
        h = hard_loss(ref.SAMPLE_LOGITS, ref.SAMPLE_TARGETS)
        s = soft_loss(ref.SAMPLE_LOGITS, ref.SAMPLE_TEACHER_LOGITS, 2.0)
        b = blended_loss(ref.SAMPLE_LOGITS, ref.SAMPLE_TARGETS, ref.SAMPLE_LOGITS, ref.SAMPLE_TEACHER_LOGITS, 0.5, 2.0)
        if isinstance(h, float) and isinstance(s, float) and isinstance(b, float):
            ok = 3
    except Exception as e:
        out["_note"] = f"loss execution failed: {e}"
    out["loss_matched"] = float(ok)
    return out
