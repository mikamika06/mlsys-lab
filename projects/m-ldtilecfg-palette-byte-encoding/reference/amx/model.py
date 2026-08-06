def amx_vs_avx512_throughput(params):
    clock_ghz = params["clock_ghz"]
    num_cores = params["num_cores"]

    amx_ops_per_cycle = params["amx_ops_per_cycle_per_core"]
    avx512_ops_per_cycle = params["avx512_ops_per_cycle_per_core"]

    amx_peak_tflops = num_cores * clock_ghz * amx_ops_per_cycle / 1000.0
    avx512_peak_tflops = num_cores * clock_ghz * avx512_ops_per_cycle / 1000.0

    derived_ceiling = amx_peak_tflops / avx512_peak_tflops

    amx_eff = params.get("amx_efficiency", 1.0)
    avx_eff = params.get("avx512_efficiency", 1.0)

    measured_amx = amx_peak_tflops * amx_eff
    measured_avx = avx512_peak_tflops * avx_eff
    measured_speedup = measured_amx / measured_avx

    return {
        "amx_peak_tflops": amx_peak_tflops,
        "avx512_peak_tflops": avx512_peak_tflops,
        "derived_ceiling": derived_ceiling,
        "measured_speedup": measured_speedup
    }


def tmul_time_share(params):
    m, n, k = params["m"], params["n"], params["k"]
    ops_per_tmul = 2 * m * n * k

    compute_cycles = ops_per_tmul / params["ops_per_cycle"]

    bytes_loaded = params.get("bytes_loaded", (m * k) + (k * n))
    bytes_stored = params.get("bytes_stored", m * n)
    total_bytes = bytes_loaded + bytes_stored

    mem_cycles = total_bytes / params["mem_bytes_per_cycle"]

    total_cycles = compute_cycles + mem_cycles

    return {
        "compute_cycles": compute_cycles,
        "mem_cycles": mem_cycles,
        "total_cycles": total_cycles,
        "tmul_compute_share": compute_cycles / total_cycles,
        "tile_io_share": mem_cycles / total_cycles
    }
