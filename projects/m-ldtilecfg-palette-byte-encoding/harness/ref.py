CONFIG_FIXTURES = [
    {
        "palette_id": 1,
        "start_row": 0,
        "tiles": {
            0: {"bytes_per_row": 64, "rows": 16},
            1: {"bytes_per_row": 64, "rows": 16},
            2: {"bytes_per_row": 64, "rows": 16}
        }
    },
    {
        "palette_id": 1,
        "start_row": 0,
        "tiles": {
            0: {"bytes_per_row": 32, "rows": 8},
            7: {"bytes_per_row": 64, "rows": 16}
        }
    },
    {
        "palette_id": 1,
        "start_row": 4,
        "tiles": {
            3: {"bytes_per_row": 16, "rows": 4}
        }
    },
    {
        "palette_id": 1,
        "start_row": 0,
        "tiles": {i: {"bytes_per_row": 64, "rows": 16} for i in range(8)}
    },
    {
        "palette_id": 1,
        "start_row": 0,
        "tiles": {}
    }
]

THROUGHPUT_FIXTURES = [
    {
        "clock_ghz": 2.0, "num_cores": 16,
        "amx_ops_per_cycle_per_core": 2048, "avx512_ops_per_cycle_per_core": 256,
        "amx_efficiency": 0.85, "avx512_efficiency": 0.90
    },
    {
        "clock_ghz": 2.5, "num_cores": 32,
        "amx_ops_per_cycle_per_core": 1024, "avx512_ops_per_cycle_per_core": 128,
        "amx_efficiency": 0.70, "avx512_efficiency": 0.85
    }
]

TIME_SHARE_FIXTURES = [
    {
        "m": 16, "n": 16, "k": 64,
        "ops_per_cycle": 2048, "mem_bytes_per_cycle": 64,
        "bytes_loaded": 2048, "bytes_stored": 512
    },
    {
        "m": 16, "n": 16, "k": 32,
        "ops_per_cycle": 1024, "mem_bytes_per_cycle": 32,
        "bytes_loaded": 1024, "bytes_stored": 256
    }
]


def encode_tilecfg(tile_specs, palette_id=1, start_row=0):
    buf = bytearray(64)
    buf[0] = palette_id & 0xFF
    buf[1] = start_row & 0xFF
    for t_id, spec in tile_specs.items():
        if 0 <= t_id <= 7:
            colsb = spec.get("bytes_per_row", 0)
            rows = spec.get("rows", 0)
            buf[16 + t_id * 2] = colsb & 0xFF
            buf[16 + t_id * 2 + 1] = (colsb >> 8) & 0xFF
            buf[48 + t_id] = rows & 0xFF
    return bytes(buf)


def decode_tilecfg(buffer):
    palette_id = buffer[0]
    start_row = buffer[1]
    tiles = {}
    for t_id in range(8):
        colsb = buffer[16 + t_id * 2] | (buffer[16 + t_id * 2 + 1] << 8)
        rows = buffer[48 + t_id]
        if colsb > 0 or rows > 0:
            tiles[t_id] = {"bytes_per_row": colsb, "rows": rows}
    return {"palette_id": palette_id, "start_row": start_row, "tiles": tiles}


def amx_vs_avx512_throughput(params):
    c, n = params["clock_ghz"], params["num_cores"]
    amx_peak = n * c * params["amx_ops_per_cycle_per_core"] / 1000.0
    avx_peak = n * c * params["avx512_ops_per_cycle_per_core"] / 1000.0
    ceiling = amx_peak / avx_peak
    speedup = (amx_peak * params.get("amx_efficiency", 1.0)) / (avx_peak * params.get("avx512_efficiency", 1.0))
    return {
        "amx_peak_tflops": amx_peak,
        "avx512_peak_tflops": avx_peak,
        "derived_ceiling": ceiling,
        "measured_speedup": speedup
    }


def tmul_time_share(params):
    m, n, k = params["m"], params["n"], params["k"]
    compute_cycles = (2 * m * n * k) / params["ops_per_cycle"]
    bytes_loaded = params.get("bytes_loaded", (m * k) + (k * n))
    bytes_stored = params.get("bytes_stored", m * n)
    mem_cycles = (bytes_loaded + bytes_stored) / params["mem_bytes_per_cycle"]
    tot = compute_cycles + mem_cycles
    return {
        "compute_cycles": compute_cycles,
        "mem_cycles": mem_cycles,
        "total_cycles": tot,
        "tmul_compute_share": compute_cycles / tot,
        "tile_io_share": mem_cycles / tot
    }
