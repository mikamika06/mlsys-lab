import ref
import numpy as np

def check(workdir):
    from kbitbug.reproduce import simulate_training_step
    np.random.seed(123)
    inputs = np.random.randn(10, 4)
    weights = np.random.randn(4, 2)
    targets = np.random.randn(10, 2)

    loss_sk, grad_sk = simulate_training_step(weights, inputs, targets, skipped_preparation=True)
    loss_ok, grad_ok = simulate_training_step(weights, inputs, targets, skipped_preparation=False)

    ref_loss_sk, ref_grad_sk = ref.simulate_training_step(weights, inputs, targets, skipped_preparation=True)
    ref_loss_ok, ref_grad_ok = ref.simulate_training_step(weights, inputs, targets, skipped_preparation=False)

    match_sk = np.allclose(grad_sk, ref_grad_sk) and np.allclose(loss_sk, ref_loss_sk)
    match_ok = np.allclose(grad_ok, ref_grad_ok) and np.allclose(loss_ok, ref_loss_ok)

    is_zero = np.all(grad_sk == 0)
    has_nonzero = not np.all(grad_ok == 0)

    out = {"gradient_behavior_matched": 1.0 if (match_sk and match_ok and is_zero and has_nonzero) else 0.0}
    if out["gradient_behavior_matched"] == 0.0:
        out["_note"] = "Simulation results or zero-gradient condition did not match reference expectations."
    return out
