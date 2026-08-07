import numpy as np
import ref


def check(workdir):
    from distill.diagnose import derive_effective_temperature, detect_mode_collapse

    rng = np.random.RandomState(42)
    history = [
        rng.randn(20, 5),
        rng.randn(20, 5) * 0.1,
        np.hstack([np.ones((20, 1)) * 15.0, rng.randn(20, 4)]),
    ]
    threshold = 0.5

    want_collapse = ref.detect_mode_collapse(history, threshold)
    try:
        got_collapse = detect_mode_collapse(history, threshold)
    except Exception as e:
        return {
            "collapse_detected": 0.0,
            "temp_shift_rel_err": 1.0,
            "_note": f"detect_mode_collapse failed: {type(e).__name__}: {str(e)[:100]}",
        }

    collapse_ok = 1.0 if list(got_collapse) == list(want_collapse) else 0.0

    teacher_logits = ref.generate_teacher_logits(num_samples=50, num_classes=5, seed=99)
    target_temp = 2.5
    alpha = 0.8

    want_temp = ref.derive_effective_temperature(teacher_logits, target_temp, alpha)
    try:
        got_temp = derive_effective_temperature(teacher_logits, target_temp, alpha)
    except Exception as e:
        return {
            "collapse_detected": collapse_ok,
            "temp_shift_rel_err": 1.0,
            "_note": f"derive_effective_temperature failed: {type(e).__name__}: {str(e)[:100]}",
        }

    temp_err = abs(got_temp - want_temp) / (abs(want_temp) + 1e-12)

    return {
        "collapse_detected": collapse_ok,
        "temp_shift_rel_err": float(temp_err),
    }
