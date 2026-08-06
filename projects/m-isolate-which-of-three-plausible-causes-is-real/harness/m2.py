"""Milestone 2 harness check."""

import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)

    out = {"accuracy": 0.0}
    try:
        from triage.classifier import triage_log_batch
    except Exception as e:
        out["_note"] = f"Failed to import triage_log_batch: {e}"
        return out

    samples, labels = ref.generate_samples(seed=456, n=100)
    try:
        preds = triage_log_batch(samples)
    except Exception as e:
        out["_note"] = f"triage_log_batch raised exception: {e}"
        return out

    if not isinstance(preds, list) or len(preds) != len(labels):
        out["_note"] = "triage_log_batch did not return a list matching input length"
        return out

    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    out["accuracy"] = float(correct) / float(len(labels))
    return out
