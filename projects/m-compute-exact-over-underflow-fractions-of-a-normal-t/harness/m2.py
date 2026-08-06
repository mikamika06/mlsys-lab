import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"classification_accuracy": 0.0, "simulation_rel_err": 1.0}

    try:
        from numdiag.classifier import classify_training_log_symptoms
        from numdiag.quantization import simulate_nf4_compounding_error
    except Exception as e:
        out["_note"] = f"Import error: {type(e).__name__}: {str(e)}"
        return out

    test_logs = [
        {"grad_norm": 0.0, "loss": 1.5, "loss_delta": -0.01, "unique_activation_ratio": 1.0},
        {"grad_norm": 15000.0, "loss": float("nan"), "is_nan_or_inf": True},
        {"grad_norm": 1e-7, "loss": 0.8, "loss_delta": 1e-8, "unique_activation_ratio": 0.9},
        {"grad_norm": 0.5, "loss": 0.2, "loss_delta": -0.02, "unique_activation_ratio": 0.02},
    ]

    expected_classes = ref.ref_classify_symptoms(test_logs)
    try:
        actual_classes = classify_training_log_symptoms(test_logs)
        correct = sum(1 for e, a in zip(expected_classes, actual_classes) if e == a)
        out["classification_accuracy"] = float(correct / len(expected_classes))
    except Exception as e:
        out["_note"] = f"Classification failed: {type(e).__name__}: {str(e)}"
        return out

    tensors = ref.get_test_tensors()
    sim_tensor = tensors["standard_normal"]

    expected_sim = ref.ref_simulate_nf4(sim_tensor, num_cycles=10)
    try:
        actual_sim = simulate_nf4_compounding_error(sim_tensor, num_cycles=10)
    except Exception as e:
        out["_note"] = f"Simulation failed: {type(e).__name__}: {str(e)}"
        return out

    mse_err = np.max(np.abs(expected_sim["mse_history"] - actual_sim["mse_history"])) / (
        np.max(np.abs(expected_sim["mse_history"])) + 1e-12
    )
    max_err_err = np.max(np.abs(expected_sim["max_err_history"] - actual_sim["max_err_history"])) / (
        np.max(np.abs(expected_sim["max_err_history"])) + 1e-12
    )

    out["simulation_rel_err"] = float(max(mse_err, max_err_err))
    return out
