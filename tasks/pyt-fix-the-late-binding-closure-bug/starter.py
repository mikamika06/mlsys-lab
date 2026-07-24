def make_multipliers():
    """Return a list of multipliers: [f_0, f_1, ..., f_4] where f_i(x) = i * x."""
    # BUG: all lambdas capture the same loop variable — fix the late-binding issue.
    return [lambda x: i * x for i in range(5)]
