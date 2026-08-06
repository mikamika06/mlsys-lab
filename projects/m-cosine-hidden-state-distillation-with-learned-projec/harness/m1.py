import numpy as np
import ref


def check(workdir):
    from distill.hidden import LearnedProjectionCosineLoss

    fixtures = ref.generate_fixtures(seed=101)
    s_state = fixtures["student_state"]
    t_state = fixtures["teacher_state"]

    ref_loss_fn = LearnedProjectionCosineLoss(student_dim=64, teacher_dim=128, seed=42)
    ref_fwd = ref_loss_fn.forward(s_state, t_state)
    ref_bwd = ref_loss_fn.backward(s_state, t_state)

    try:
        user_loss_fn = LearnedProjectionCosineLoss(student_dim=64, teacher_dim=128, seed=42)
        user_fwd = user_loss_fn.forward(s_state, t_state)
        user_bwd = user_loss_fn.backward(s_state, t_state)
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"Exception during execution: {e}"}

    fwd_err = abs(user_fwd - ref_fwd) / (abs(ref_fwd) + 1e-12)
    bwd_err = np.max(np.abs(user_bwd - ref_bwd)) / (np.max(np.abs(ref_bwd)) + 1e-12)
    total_err = float(max(fwd_err, bwd_err))

    return {"rel_err": total_err}
