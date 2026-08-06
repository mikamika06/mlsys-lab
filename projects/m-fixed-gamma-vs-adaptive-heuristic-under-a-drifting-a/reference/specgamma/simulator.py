def run_simulation(accept_flags: list[bool], controller, cost_ratio: float = 0.05) -> dict:
    pos = 0
    total_tokens = 0
    total_cost = 0.0
    steps = 0
    n = len(accept_flags)

    while pos < n:
        gamma = controller.get_gamma()
        accepted = 0
        for k in range(gamma):
            idx = pos + k
            if idx < n and accept_flags[idx]:
                accepted += 1
            else:
                break

        tokens_produced = accepted + 1
        pos += tokens_produced
        total_tokens += tokens_produced
        step_cost = 1.0 + gamma * cost_ratio
        total_cost += step_cost
        steps += 1

        controller.update(accepted, gamma)

    throughput = total_tokens / total_cost if total_cost > 0 else 0.0
    return {
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "throughput": throughput,
        "steps": steps,
    }
