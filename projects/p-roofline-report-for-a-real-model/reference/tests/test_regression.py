import sys
sys.path.insert(0, ".")
from roofline.models import Hardware, KernelRun
from roofline.analysis import generate_report

def test_report_is_sorted():
    hw = Hardware(peak_gflops=19500.0, peak_gbps=1555.0)
    kernels = [
        KernelRun("attn", 500000000, 200000000, 5.0),
        KernelRun("mlp", 1000000000, 50000000, 15.0),
        KernelRun("norm", 1000000, 10000000, 2.0)
    ]
    report = generate_report(hw, kernels)

    for i in range(len(report) - 1):
        assert report[i]["saved_ms"] >= report[i+1]["saved_ms"], "Report is not sorted by saved_ms descending"

def test_no_negative_saved_time():
    hw = Hardware(peak_gflops=19500.0, peak_gbps=1555.0)
    kernels = [
        KernelRun("fast_kernel", 500000, 200000, 0.00001)
    ]
    report = generate_report(hw, kernels)
    assert report[0]["saved_ms"] >= 0.0, "Saved time cannot be negative"
