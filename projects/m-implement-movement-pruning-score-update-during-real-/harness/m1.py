import ref
import numpy as np


def check(workdir):
    from movement.prune import update_movement_scores
    data = ref.get_test_cases()

    want = data["scores"] + data["weights"] * data["grads"] * data["lr"]
    try:
        got = update_movement_scores(data["scores"], data["weights"], data["grads"], data["lr"])
    except Exception as e:
        return {"scores_matched": 0.0, "_note": f"raised exception: {e}"}

    if got is None:
        return {"scores_matched": 0.0, "_note": "returned None"}

    diff = np.max(np.abs(np.array(got) - want))
    matched = 1.0 if diff < 1e-5 else 0.0
    out = {"scores_matched": matched}
    if matched == 0.0:
        out["_note"] = f"max diff: {diff}"
    return out
