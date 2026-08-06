import ref
from avx512_mechanics.downclock import simulate_execution as ref_sim
from avx512_mechanics.vnni import analyze_vnni_vs_fallback as ref_vnni


def check(workdir):
    from avx512_mechanics.downclock import simulate_execution
    from avx512_mechanics.vnni import analyze_vnni_vs_fallback

    out = {"downclock_rel_err": 1.0, "vnni_rel_err": 1.0}

    sim_cases = ref.get_simulation_cases()
    max_sim_err = 0.0
    for case in sim_cases:
        want = ref_sim(**case)
        got = simulate_execution(**case)
        err = abs(got["total_time_us"] - want["total_time_us"]) / max(
            1e-9, want["total_time_us"]
        )
        if err > max_sim_err:
            max_sim_err = err

    out["downclock_rel_err"] = float(max_sim_err)

    vnni_cases = ref.get_vnni_cases()
    max_vnni_err = 0.0
    for case in vnni_cases:
        want = ref_vnni(**case)
        got = analyze_vnni_vs_fallback(**case)
        err = abs(got["measured_speedup"] - want["measured_speedup"]) / max(
            1e-9, want["measured_speedup"]
        )
        if err > max_vnni_err:
            max_vnni_err = err

    out["vnni_rel_err"] = float(max_vnni_err)
    return out
