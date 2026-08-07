import numpy as np
import ref


def check(workdir):
    from distill.gap import simulate_capacity_loss_floor

    teacher_logits = ref.generate_teacher_logits(num_samples=80, num_classes=8, seed=123)
    rank_constraint = 3
    temperature_grid = np.array([0.5, 1.0, 2.0, 4.0])

    want = ref.simulate_capacity_loss_floor(teacher_logits, rank_constraint, temperature_grid)
    try:
        got = simulate_capacity_loss_floor(teacher_logits, rank_constraint, temperature_grid)
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"Execution failed: {type(e).__name__}: {str(e)[:100]}"}

    if not isinstance(got, np.ndarray) or got.shape != want.shape:
        return {"rel_err": 1.0, "_note": f"Output shape mismatch: got {getattr(got, 'shape', None)}, expected {want.shape}"}

    err = np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-12)
    return {"rel_err": float(err)}
