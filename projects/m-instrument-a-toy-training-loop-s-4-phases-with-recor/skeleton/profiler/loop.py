def profile_training_loop(step_fn, num_steps=5):
    """Run step_fn under profiler and report percentage time for 4 phases."""
    raise NotImplementedError


def compute_uncovered_time_pct(step_fn, num_steps=5):
    """Compute percentage of total time spent outside record_function blocks."""
    raise NotImplementedError
