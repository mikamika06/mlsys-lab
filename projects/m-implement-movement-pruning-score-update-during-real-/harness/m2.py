import ref
import numpy as np


def check(workdir):
    from movement.prune import reconstruct_trajectory
    data = ref.get_test_cases()

    scores = np.zeros(data["weight_series"][0].shape, dtype=np.float32)
    for w, g in zip(data["weight_series"], data["grad_series"]):
        scores += w * g * data["lr"]
    want = scores

    try:
        got = reconstruct_trajectory(data["weight_series"], data["grad_series"], data["lr"])
    except Exception as e:
        return {"trajectory_error": 999.0, "_note": f"raised exception: {e}"}

    if got is None:
        return {"trajectory_error": 999.0, "_note": "returned None"}

    err = float(np.max(np.abs(np.array(got) - want)))
    return {"trajectory_error": err}
