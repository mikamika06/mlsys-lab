import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from moe.router import simulate_step_time
    except Exception:
        return {
            "step_time_model_ok": 0.0,
            "imbalance_penalty_captured": 0.0,
        }

    balanced = np.array([50, 50, 50, 50, 50, 50, 50, 50])
    imbalanced = np.array([200, 150, 10, 10, 10, 10, 5, 5])

    ref_t_bal = ref.get_ref_step_time(balanced, capacity=100)
    ref_t_imb = ref.get_ref_step_time(imbalanced, capacity=100)

    try:
        l_t_bal = simulate_step_time(balanced, capacity_per_expert=100)
        l_t_imb = simulate_step_time(imbalanced, capacity_per_expert=100)
    except Exception:
        return {
            "step_time_model_ok": 0.0,
            "imbalance_penalty_captured": 0.0,
        }

    model_ok = (
        1.0
        if (abs(l_t_bal - ref_t_bal) < 1e-3 and abs(l_t_imb - ref_t_imb) < 1e-3)
        else 0.0
    )
    penalty_ok = 1.0 if l_t_imb > 1.5 * l_t_bal else 0.0

    return {
        "step_time_model_ok": model_ok,
        "imbalance_penalty_captured": penalty_ok,
    }
