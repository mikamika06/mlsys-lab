def simulate_trajectory(overflows, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
    raise NotImplementedError


def identify_skipped_steps(overflows):
    raise NotImplementedError


def next_doubling_step(overflows, growth_interval=2000):
    raise NotImplementedError
