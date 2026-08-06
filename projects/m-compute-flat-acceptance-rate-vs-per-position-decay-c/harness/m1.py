import numpy as np
import ref


def check(workdir):
    out = {"flat_rate_rel_err": 1.0, "decay_curve_rel_err": 1.0}
    try:
        from spec.decay import compute_acceptance_metrics
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    traces = ref.generate_trace_data()
    want_flat, want_decay = ref.compute_acceptance_metrics(traces)

    try:
        got_flat, got_decay = compute_acceptance_metrics(traces)
    except Exception as e:
        out["_note"] = f"Execution error: {e}"
        return out

    flat_err = abs(got_flat - want_flat) / (abs(want_flat) + 1e-12)

    got_decay = np.asarray(got_decay, dtype=np.float64)
    want_decay = np.asarray(want_decay, dtype=np.float64)

    if got_decay.shape != want_decay.shape:
        out["_note"] = f"Shape mismatch: got {got_decay.shape}, want {want_decay.shape}"
        return out

    decay_err = float(np.max(np.abs(got_decay - want_decay) / (np.abs(want_decay) + 1e-12)))

    out["flat_rate_rel_err"] = float(flat_err)
    out["decay_curve_rel_err"] = decay_err
    return out
