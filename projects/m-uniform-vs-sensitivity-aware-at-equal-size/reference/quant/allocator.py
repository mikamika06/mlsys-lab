from quant.profile import should_exclude


def total_model_bytes(modules: list[dict], bit_assignment: dict[str, int]) -> float:
    """Compute the total model footprint in bytes given a bit assignment."""
    total = 0.0
    for mod in modules:
        bits = bit_assignment[mod["name"]]
        total += mod["num_params"] * (bits / 8.0)
    return total


def uniform_allocation(
    modules: list[dict],
    exclude_patterns: list[str],
    target_bits: int,
    default_fp_bits: int = 16,
) -> dict[str, int]:
    """Produce a uniform bit allocation while keeping excluded modules at default FP bits."""
    assignment = {}
    for mod in modules:
        if should_exclude(mod["name"], exclude_patterns):
            assignment[mod["name"]] = default_fp_bits
        else:
            assignment[mod["name"]] = target_bits
    return assignment


def greedy_allocation(
    modules: list[dict],
    sensitivity_profile: dict[str, dict[int, float]],
    exclude_patterns: list[str],
    max_bytes: float,
    candidate_bits: list[int],
    default_fp_bits: int = 16,
) -> dict[str, int]:
    """Perform greedy bit allocation based on marginal error reduction per byte."""
    bits_sorted = sorted(candidate_bits)
    assignment = {}
    for mod in modules:
        if should_exclude(mod["name"], exclude_patterns):
            assignment[mod["name"]] = default_fp_bits
        else:
            assignment[mod["name"]] = bits_sorted[0]

    while True:
        best_mod = None
        best_next_bit = None
        best_efficiency = -1.0

        current_bytes = total_model_bytes(modules, assignment)

        for mod in modules:
            m_name = mod["name"]
            if should_exclude(m_name, exclude_patterns):
                continue

            current_b = assignment[m_name]
            higher_bits = [b for b in bits_sorted if b > current_b]
            if not higher_bits:
                continue

            next_b = higher_bits[0]
            delta_bits = next_b - current_b
            added_bytes = mod["num_params"] * (delta_bits / 8.0)

            if current_bytes + added_bytes > max_bytes + 1e-7:
                continue

            current_err = sensitivity_profile[m_name][current_b]
            next_err = sensitivity_profile[m_name][next_b]
            err_reduction = current_err - next_err

            efficiency = err_reduction / added_bytes if added_bytes > 0 else 0.0
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_mod = m_name
                best_next_bit = next_b

        if best_mod is None or best_efficiency <= 0.0:
            break

        assignment[best_mod] = best_next_bit

    return assignment


def optimal_allocation(
    modules: list[dict],
    sensitivity_profile: dict[str, dict[int, float]],
    exclude_patterns: list[str],
    max_bytes: float,
    candidate_bits: list[int],
    default_fp_bits: int = 16,
) -> dict[str, int]:
    """Find optimal bit allocation minimizing overall error under max_bytes using dynamic programming."""
    quant_modules = [m for m in modules if not should_exclude(m["name"], exclude_patterns)]
    fixed_bytes = sum(
        m["num_params"] * (default_fp_bits / 8.0)
        for m in modules
        if should_exclude(m["name"], exclude_patterns)
    )

    available_bytes = max_bytes - fixed_bytes
    if available_bytes < 0:
        assignment = {}
        for m in modules:
            if should_exclude(m["name"], exclude_patterns):
                assignment[m["name"]] = default_fp_bits
            else:
                assignment[m["name"]] = min(candidate_bits)
        return assignment

    granularity = 8
    max_units = int(available_bytes * granularity + 1e-7)

    dp = {0: (0.0, {})}

    for mod in quant_modules:
        m_name = mod["name"]
        num_p = mod["num_params"]
        next_dp = {}

        for u_bytes, (err_sum, assign) in dp.items():
            for b in candidate_bits:
                cost_u = int(round(num_p * (b / 8.0) * granularity))
                new_u = u_bytes + cost_u
                if new_u > max_units:
                    continue
                new_err = err_sum + sensitivity_profile[m_name][b]

                if new_u not in next_dp or new_err < next_dp[new_u][0] - 1e-9:
                    new_assign = dict(assign)
                    new_assign[m_name] = b
                    next_dp[new_u] = (new_err, new_assign)

        dp = next_dp

    if not dp:
        return greedy_allocation(
            modules,
            sensitivity_profile,
            exclude_patterns,
            max_bytes,
            candidate_bits,
            default_fp_bits,
        )

    best_u = min(dp.keys(), key=lambda u: (dp[u][0], u))
    opt_quant = dp[best_u][1]

    full_assignment = {}
    for m in modules:
        m_name = m["name"]
        if should_exclude(m_name, exclude_patterns):
            full_assignment[m_name] = default_fp_bits
        else:
            full_assignment[m_name] = opt_quant[m_name]

    return full_assignment


def construct_greedy_failure_case() -> tuple[list[dict], dict[str, dict[int, float]], float]:
    """Construct a synthetic problem instance where greedy allocation fails to find the optimum."""
    modules = [
        {"name": "layer_0", "num_params": 800},
        {"name": "layer_1", "num_params": 800},
    ]

    profile = {
        "layer_0": {2: 100.0, 4: 90.0, 8: 10.0},
        "layer_1": {2: 100.0, 4: 80.0, 8: 75.0},
    }

    max_bytes = 800 * (2 / 8.0) + 800 * (8 / 8.0)
    return modules, profile, max_bytes
