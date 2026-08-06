"""Collate profiling and budget verification utilities."""

def profile_collate(collate_fn, sample_generator, batch_sizes):
    """Profile collate_fn execution time across various batch sizes."""
    raise NotImplementedError


def evaluate_budget(profile_results, max_budget_ms_per_batch, target_throughput_samples_sec):
    """Evaluate whether the collate profile satisfies budget and throughput rules."""
    raise NotImplementedError
