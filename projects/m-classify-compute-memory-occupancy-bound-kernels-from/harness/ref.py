import random

def get_ncu_samples():
    return [
        (
            "sm__throughput.avg.pct_of_peak_sustained_elapsed 85.5\n"
            "dram__throughput.avg.pct_of_peak_sustained_elapsed 30.2\n"
            "sm__warps_active.avg.pct_of_peak_sustained_active 75.0\n",
            "compute-bound"
        ),
        (
            "sm__throughput.avg.pct_of_peak_sustained_elapsed 20.1\n"
            "dram__throughput.avg.pct_of_peak_sustained_elapsed 88.4\n"
            "sm__warps_active.avg.pct_of_peak_sustained_active 80.0\n",
            "memory-bound"
        ),
        (
            "sm__throughput.avg.pct_of_peak_sustained_elapsed 30.0\n"
            "dram__throughput.avg.pct_of_peak_sustained_elapsed 25.0\n"
            "sm__warps_active.avg.pct_of_peak_sustained_active 25.0\n",
            "occupancy-bound"
        ),
        (
            "sm__throughput.avg.pct_of_peak_sustained_elapsed 90.0\n"
            "dram__throughput.avg.pct_of_peak_sustained_elapsed 40.0\n"
            "sm__warps_active.avg.pct_of_peak_sustained_active 95.0\n",
            "compute-bound"
        ),
        (
            "sm__throughput.avg.pct_of_peak_sustained_elapsed 15.0\n"
            "dram__throughput.avg.pct_of_peak_sustained_elapsed 92.5\n"
            "sm__warps_active.avg.pct_of_peak_sustained_active 60.0\n",
            "memory-bound"
        ),
    ]

def get_proton_sample():
    lines = [
        "region_attention,500.0",
        "region_gemm,300.0",
        "region_layernorm,200.0",
    ]
    expected = {
        "region_attention": 50.0,
        "region_gemm": 30.0,
        "region_layernorm": 20.0,
    }
    return lines, expected

def get_torch_trace_sample():
    trace = {
        "dur": 2000.0,
        "args": {
            "block X": 256,
            "block Y": 1,
            "block Z": 1,
            "grid X": 128,
            "grid Y": 1,
            "grid Z": 1,
            "flops": 2000000000000.0,
        }
    }
    expected = {
        "block_size": [256, 1, 1],
        "grid_size": [128, 1, 1],
        "tflops": 1.0,
    }
    return trace, expected
