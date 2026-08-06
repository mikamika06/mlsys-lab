from avx512_mechanics.downclock import simulate_execution


def derived_mac_per_cycle(is_vnni: bool, vector_width: int) -> float:
    """Return theoretical MAC operations per cycle."""
    if is_vnni:
        if vector_width == 512:
            return 128.0
        if vector_width == 256:
            return 64.0
    else:
        if vector_width == 512:
            return 32.0
        if vector_width == 256:
            return 16.0
    raise ValueError(f"Unsupported config: VNNI={is_vnni}, width={vector_width}")


def analyze_vnni_vs_fallback(
    num_mac_ops: int,
    vnni_vector_width: int = 512,
    fallback_vector_width: int = 256,
    base_freq_ghz: float = 3.0,
    recovery_cycles: int = 50000,
) -> dict:
    """Analyze VNNI speedup vs fallback including downclocking penalty."""
    vnni_mac_cycle = derived_mac_per_cycle(True, vnni_vector_width)
    fallback_mac_cycle = derived_mac_per_cycle(False, fallback_vector_width)

    derived_ratio = vnni_mac_cycle / fallback_mac_cycle

    vnni_nominal_cycles = int(num_mac_ops / vnni_mac_cycle)
    fallback_nominal_cycles = int(num_mac_ops / fallback_mac_cycle)

    vnni_instr = (
        "vpdpbusd zmm1, zmm2, zmm3"
        if vnni_vector_width == 512
        else "vpdpbusd ymm1, ymm2, ymm3"
    )
    fallback_instr = (
        "vpmaddwd ymm1, ymm2, ymm3"
        if fallback_vector_width == 256
        else "vpmaddwd zmm1, zmm2, zmm3"
    )

    vnni_sim = simulate_execution(
        [(vnni_instr, vnni_nominal_cycles)], base_freq_ghz, recovery_cycles
    )
    fallback_sim = simulate_execution(
        [(fallback_instr, fallback_nominal_cycles)], base_freq_ghz, recovery_cycles
    )

    vnni_time = vnni_sim["total_time_us"]
    fallback_time = fallback_sim["total_time_us"]

    measured_speedup = fallback_time / vnni_time
    downclock_penalty = measured_speedup / derived_ratio

    return {
        "derived_mac_ratio": derived_ratio,
        "measured_speedup": measured_speedup,
        "downclock_penalty": downclock_penalty,
        "vnni_time_us": vnni_time,
        "fallback_time_us": fallback_time,
    }
