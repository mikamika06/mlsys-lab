"""Batch log triage classifier."""

from triage.isolate import isolate_root_cause


def triage_log_batch(samples):
    """Classify a list of diagnostic log samples and return a list of predicted cause strings."""
    return [isolate_root_cause(s) for s in samples]
