from avx512_mechanics.downclock import simulate_execution


def test_downclock_recovery_hysteresis():
    stream = [("vpdpbusd zmm1, zmm2, zmm3", 100), ("add rax, rbx", 50000)]
    res_with_recovery = simulate_execution(
        stream, base_freq_ghz=3.0, recovery_cycles=50000
    )
    res_no_recovery = simulate_execution(
        stream, base_freq_ghz=3.0, recovery_cycles=0
    )

    assert res_with_recovery["total_time_us"] > res_no_recovery["total_time_us"]
    assert (
        res_with_recovery["tier_durations_us"]["L2"]
        > res_no_recovery["tier_durations_us"]["L2"]
    )
