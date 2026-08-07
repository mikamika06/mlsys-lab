import sys
sys.path.insert(0, ".")
from triton_profiler.interpret import run_interpreted_kernel
from triton_profiler.parser import compute_region_times
from triton_profiler.autotune import select_fastest_config


def test_interpret_basic():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    res = run_interpreted_kernel(x, y, 2)
    assert len(res["per_block_times"]) == 2
    assert abs(res["output"][0] - 11.0) < 1e-6


def test_parser_sum():
    tree = {
        "name": "root",
        "duration": 100.0,
        "children": [
            {"name": "kernel_a", "duration": 40.0, "children": []},
            {"name": "kernel_b", "duration": 60.0, "children": []}
        ]
    }
    pcts = compute_region_times(tree)
    assert abs(pcts["kernel_a"] - 40.0) < 1e-5
    assert abs(pcts["kernel_b"] - 60.0) < 1e-5


def test_autotune_selection():
    sweeps = [
        {"config": {"BLOCK_SIZE": 32}, "latency": 15.2},
        {"config": {"BLOCK_SIZE": 64}, "latency": 10.1},
        {"config": {"BLOCK_SIZE": 128}, "latency": 22.4}
    ]
    best = select_fastest_config(sweeps)
    assert best == {"BLOCK_SIZE": 64}
