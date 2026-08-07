def moe_all_to_all_dispatch(X: list[list[float]], router_logits: list[list[float]], expert_weight: list[list[list[float]]], num_devices: int):
    """Simulate expert-parallel MoE dispatch: route -> all-to-all -> expert
    compute -> all-to-all back.

    See task.md for the exact routing/placement rule.
    """
    raise NotImplementedError('your code here')
