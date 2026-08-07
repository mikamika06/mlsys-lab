def train_run(seed, deterministic):
    """Run training step."""
    raise NotImplementedError

def measure_cost():
    """Measure deterministic overhead."""
    raise NotImplementedError

def is_deterministic():
    """Return flag status."""
    raise NotImplementedError
