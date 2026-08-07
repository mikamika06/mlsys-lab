import sys

sys.path.insert(0, ".")
from roofline.intensity import compute_kernel_intensity, aggregate_profile
from roofline.model import roofline_ceiling, classify_kernel
from roofline.analysis import generate_prioritized_report


def test_compute_intensity_nonnegative():
    assert compute_kernel_intensity(1000, 500) == 2.0
    assert compute_kernel_intensity(1000, 0) == 0.0


def test_roofline_ceiling_bounds():
    hw = {"peak_flops_per_sec": 100e12, "peak_bandwidth_bytes_sec": 2e12}
    assert roofline_ceiling(10.0, hw) == 20e12
    assert roofline_ceiling(100.0, hw) == 100e12


def test_classify_kernel():
    hw = {"peak_flops_per_sec": 100e12, "peak_bandwidth_bytes_sec": 2e12}
    assert classify_kernel(10.0, hw) == "memory_bound"
    assert classify_kernel(100.0, hw) == "compute_bound"


def test_prioritized_report_ordering():
    hw = {"peak_flops_per_sec": 100e12, "peak_bandwidth_bytes_sec": 2e12}
    records = [
        {"name": "k1", "flops": 1e9, "bytes": 1e8, "time_us": 1000.0},
        {"name": "k2", "flops": 1e11, "bytes": 1e7, "time_us": 500.0},
    ]
    agg = aggregate_profile(records)
    rep = generate_prioritized_report(agg, hw)
    assert len(rep) == 2
    assert rep[0]["potential_savings_us"] >= rep[1]["potential_savings_us"]
