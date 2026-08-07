import ref
import torch

def check(workdir):
    from chkpt.reentrant import run_reentrant_test
    inputs = ref.get_test_inputs()
    model = ref.get_test_model()

    try:
        success_false, _ = run_reentrant_test(model, inputs, use_reentrant=False)
        success_true, _ = run_reentrant_test(model, inputs, use_reentrant=True)

        verified = (success_false is True)
        return {"reentrant_behavior_verified": 1.0 if verified else 0.0}
    except Exception as e:
        return {"reentrant_behavior_verified": 0.0, "_note": f"failed: {type(e).__name__}: {e}"}
