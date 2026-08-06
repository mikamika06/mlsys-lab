import sys
sys.path.insert(0, ".")
from solparser.parser import extract_sol_metrics
from solparser.diff import diff_basic_full
from solparser.safety import check_replay_trustworthiness

SAMPLE_CSV = """ID,Kernel Name,Metric Name,Metric Value,Section Name,Warning
1,gemm_kernel,gpu__time_duration.sum,1234.5,SpeedOfLight,
2,attention_kernel,gpu__time_duration.sum,5678.0,SpeedOfLight,
3,gemm_kernel,sm__throughput.avg.pct_of_peak_sustained_elapsed,85.5,SpeedOfLight,
4,attention_kernel,sm__throughput.avg.pct_of_peak_sustained_elapsed,90.2,SpeedOfLight,
"""

FULL_CSV = """ID,Kernel Name,Metric Name,Metric Value,Section Name,Warning
1,gemm_kernel,gpu__time_duration.sum,1234.5,SpeedOfLight,
2,gemm_kernel,sm__throughput.avg.pct_of_peak_sustained_elapsed,85.5,SpeedOfLight,
3,gemm_kernel,dram__throughput.avg.pct_of_peak_sustained_elapsed,45.0,MemoryWorkloadAnalysis,
"""

WARNING_CSV = """ID,Kernel Name,Metric Name,Metric Value,Section Name,Warning
1,unstable_kernel,gpu__time_duration.sum,999.0,SpeedOfLight,kernel replay mismatch detected
"""

def test_extract_valid_metrics():
    res = extract_sol_metrics(SAMPLE_CSV, ["gemm_kernel"], "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    assert res["gemm_kernel"] == 85.5

def test_diff_basic_full_sections():
    diffs = diff_basic_full(SAMPLE_CSV, FULL_CSV)
    assert "MemoryWorkloadAnalysis" in diffs

def test_safety_check_detects_replay_mismatch():
    bad = check_replay_trustworthiness(WARNING_CSV)
    assert "unstable_kernel" in bad
