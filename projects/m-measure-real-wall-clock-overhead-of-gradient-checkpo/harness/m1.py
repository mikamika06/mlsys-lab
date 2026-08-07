import ref
import torch

def check(workdir):
    from chkpt.measure import measure_checkpoint_overhead
    inputs = ref.get_test_inputs()
    model = ref.get_test_model()

    try:
        t = measure_checkpoint_overhead(model, inputs)
        valid = isinstance(t, (int, float)) and t > 0.0
        return {"overhead_measured": 1.0 if valid else 0.0}
    except Exception as e:
        return {"overhead_measured": 0.0, "_note": f"failed: {type(e).__name__}: {e}"}
