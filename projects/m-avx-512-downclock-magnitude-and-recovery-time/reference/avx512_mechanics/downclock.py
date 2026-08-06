from avx512_mechanics.classifier import classify_instruction

TIER_FACTORS = {"L0": 1.0, "L1": 0.85, "L2": 0.70}


def simulate_execution(
    stream: list[tuple[str, int]],
    base_freq_ghz: float,
    recovery_cycles: int,
) -> dict:
    """Simulate instruction stream execution under downclock recovery rules."""
    r1 = 0
    r2 = 0
    total_time_s = 0.0
    total_effective_cycles = 0.0
    tier_durations = {"L0": 0.0, "L1": 0.0, "L2": 0.0}

    for instr, cycles in stream:
        tier = classify_instruction(instr)
        if tier == "L2":
            r2 = recovery_cycles
            r1 = max(r1, recovery_cycles)
        elif tier == "L1":
            r1 = max(r1, recovery_cycles)

        rem = cycles
        while rem > 0:
            if r2 > 0:
                curr_tier = "L2"
                step = min(rem, r2)
            elif r1 > 0:
                curr_tier = "L1"
                step = min(rem, r1)
            else:
                curr_tier = "L0"
                step = rem

            factor = TIER_FACTORS[curr_tier]
            eff_cycles = step / factor
            dt = eff_cycles / (base_freq_ghz * 1e9)

            total_effective_cycles += eff_cycles
            total_time_s += dt
            tier_durations[curr_tier] += dt

            r2 = max(0, r2 - step)
            r1 = max(0, r1 - step)
            rem -= step

    return {
        "total_time_us": total_time_s * 1e6,
        "effective_cycles": total_effective_cycles,
        "tier_durations_us": {k: v * 1e6 for k, v in tier_durations.items()},
    }
