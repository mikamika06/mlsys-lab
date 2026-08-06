import sys
import time
import torch
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"timing_correct": 0.0, "rel_err": 0.0}
    try:
        from mpsbench.bench import time_execution
        from mpsbench.precision import compare_precision

        def work():
            time.sleep(0.02)
            return torch.tensor([1.0])

        _, elapsed = time_execution(work, "cpu")
        if elapsed >= 0.015:
            out["timing_correct"] = 1.0
        else:
            out["_note"] = f"Timer failed to capture delay, got {elapsed:.4f}s"

        a, b = ref.make_test_tensors()
        fn = lambda x, y: x @ y
        _, _, got_err = compare_precision(a, b, fn)
        want_err = ref.compute_ref_rel_err(a, b, fn)

        if abs(got_err - want_err) <= 1e-4:
            out["rel_err"] = 1.0
        else:
            out["_note"] = f"Relative error mismatch: got {got_err}, expected {want_err}"
    except Exception as e:
        out["_note"] = f"Execution failed: {type(e).__name__}: {e}"
    return out
