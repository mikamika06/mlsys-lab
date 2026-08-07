import numpy as np
from gradaccum.accumulator import SimpleModel, run_correct_accumulation


def test_accumulation_includes_all_micro_batches():
    weights = {"w": np.array([1.0, 2.0])}
    model = SimpleModel(weights)

    mb_grads = [
        {"w": np.array([1.0, 0.0])},
        {"w": np.array([0.0, 1.0])},
        {"w": np.array([2.0, 2.0])},
        {"w": np.array([1.0, 1.0])},
    ]
    accum_steps = 4
    lr = 0.0

    grads = run_correct_accumulation(model, mb_grads, accum_steps, lr)
    assert len(grads) == 1

    expected_avg = np.array([1.0, 1.0])
    actual_grad = grads[0]["w"]

    assert np.allclose(actual_grad, expected_avg), f"Expected {expected_avg}, got {actual_grad}"


def test_single_micro_batch_not_equal_to_accumulated():
    weights = {"w": np.array([1.0, 2.0])}
    model = SimpleModel(weights)

    mb_grads = [
        {"w": np.array([4.0, 0.0])},
        {"w": np.array([0.0, 4.0])},
    ]
    accum_steps = 2
    lr = 0.0

    grads = run_correct_accumulation(model, mb_grads, accum_steps, lr)
    actual_grad = grads[0]["w"]

    last_mb_only_scaled = np.array([0.0, 2.0])
    assert not np.allclose(actual_grad, last_mb_only_scaled), "Gradient appears to only contain the last micro-batch"
