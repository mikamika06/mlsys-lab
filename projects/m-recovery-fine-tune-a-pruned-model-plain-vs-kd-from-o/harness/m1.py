import numpy as np
import ref


def check(workdir):
    from recovery.trainer import compute_kd_loss, compute_plain_loss

    X, y, teacher, student, accuracies, baseline, pruned = ref.generate_fixture()
    out = {"loss_matched": 0.0}
    try:
        p_got = compute_plain_loss(student, X, y)
        p_want = ref.ref_plain_loss(student, X, y)
        k_got = compute_kd_loss(student, teacher, X, temperature=2.0, alpha=0.5)
        k_want = ref.ref_kd_loss(student, teacher, X, temperature=2.0, alpha=0.5)
        if np.isclose(p_got, p_want, atol=1e-5) and np.isclose(k_got, k_want, atol=1e-5):
            out["loss_matched"] = 1.0
        else:
            out["_note"] = f"loss mismatch: plain got {p_got}, want {p_want}; kd got {k_got}, want {k_want}"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)[:100]}"
    return out
