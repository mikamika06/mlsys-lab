import sys
sys.path.insert(0, ".")

from amx.config import encode_tilecfg, decode_tilecfg
from amx.model import amx_vs_avx512_throughput, tmul_time_share


def test_tilecfg_size_and_header():
    specs = {0: {"bytes_per_row": 64, "rows": 16}}
    buf = encode_tilecfg(specs, palette_id=1, start_row=0)
    assert len(buf) == 64, f"Expected 64 bytes, got {len(buf)}"
    decoded = decode_tilecfg(buf)
    assert decoded["palette_id"] == 1
    assert decoded["start_row"] == 0
    assert decoded["tiles"][0] == {"bytes_per_row": 64, "rows": 16}


def test_throughput_ceiling_monotonicity():
    p = {
        "clock_ghz": 2.0,
        "num_cores": 8,
        "amx_ops_per_cycle_per_core": 2048,
        "avx512_ops_per_cycle_per_core": 256,
        "amx_efficiency": 0.8,
        "avx512_efficiency": 0.9
    }
    res = amx_vs_avx512_throughput(p)
    assert res["derived_ceiling"] == 8.0
    assert res["measured_speedup"] <= res["derived_ceiling"]


def test_tmul_time_share_bounds():
    p = {
        "m": 16, "n": 16, "k": 32,
        "ops_per_cycle": 1024,
        "mem_bytes_per_cycle": 32,
        "bytes_loaded": 1536,
        "bytes_stored": 1024
    }
    res = tmul_time_share(p)
    tot = res["tmul_compute_share"] + res["tile_io_share"]
    assert abs(tot - 1.0) < 1e-6
    assert 0.0 <= res["tmul_compute_share"] <= 1.0
