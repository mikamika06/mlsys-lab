def uniform_alloc(profile, exclude, budget, base_bits=16):
    """
    profile: list of dicts {"name": str, "params": int, "sens": dict[int, float]}
    exclude: list of str
    budget: int
    base_bits: int
    """
    raise NotImplementedError


def optimal_alloc(profile, exclude, budget, base_bits=16):
    """
    Returns dict of {layer_name: bits} minimizing sum of sensitivities.
    """
    raise NotImplementedError


def greedy_alloc(profile, exclude, budget, base_bits=16):
    """
    Start non-excluded at max bits. Repeatedly downgrade the layer with minimum
    (delta_sens / delta_bits). Break ties by layer name lexicographically.
    """
    raise NotImplementedError


def find_greedy_counterexample():
    """
    Returns (profile, exclude, budget).
    """
    raise NotImplementedError
