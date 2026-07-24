def make_multipliers():
    """Return a list of multipliers: [f_0, f_1, ..., f_4] where f_i(x) = i * x."""
    return [lambda x, i=i: i * x for i in range(5)]
