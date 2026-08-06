"""Milestone 1 harness check."""

import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)

    out = {"samples_correct": 0.0}
    try:
        from triage.isolate import isolate_root_cause
    except Exception as e:
        out["_note"] = f"Failed to import isolate_root_cause: {e}"
        return out

    samples, labels = ref.generate_samples(seed=123, n=10)
    correct = 0
    for sample, expected in zip(samples, labels):
        try:
            got = isolate_root_cause(sample)
            if got == expected:
                correct += 1
        except Exception as e:
            out["_note"] = f"isolate_root_cause raised exception: {e}"
            return out

    out["samples_correct"] = float(correct)
    return out
