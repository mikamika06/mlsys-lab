def step_loop(params, states, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    """
    Performs one Adam step using per-parameter loop.
    params: list of dicts with 'param' (np.ndarray) and 'grad' (np.ndarray)
    states: list of dicts with 'exp_avg', 'exp_avg_sq', 'step'
    """
    raise NotImplementedError


def step_foreach(params, states, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    """
    Performs one Adam step grouping tensors by shape/dtype using batched operations.
    params: list of dicts with 'param', 'grad', 'dtype', 'device'
    states: list of dicts with 'exp_avg', 'exp_avg_sq', 'step'
    """
    raise NotImplementedError


def step_fused(params, states, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    """
    Performs one Adam step applying a single fused kernel logic per parameter.
    params: list of dicts with 'param', 'grad'
    states: list of dicts with 'exp_avg', 'exp_avg_sq', 'step'
    """
    raise NotImplementedError
