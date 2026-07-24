def zero_stage_sharding(stage: int) -> tuple[bool, bool, bool]:
    """
    Return a tuple (params, grads, optimizer) indicating whether each component
    is sharded at the given ZeRO stage.

    Parameters
    ----------
    stage : int
        Zero‑based stage number; must be 0, 1, 2 or 3.

    Returns
    -------
    tuple[bool, bool, bool]
        Booleans in the order (params_sharded, grads_sharded, optimizer_sharded).
    """
    if not isinstance(stage, int):
        raise TypeError("stage must be an integer")
    if stage < 0 or stage > 3:
        raise ValueError("stage must be between 0 and 3 inclusive")

    params = [False, False, False, True]
    grads  = [False, False, True,  True]
    optimizer = [False, True,  True,  True]

    return (params[stage], grads[stage], optimizer[stage])
