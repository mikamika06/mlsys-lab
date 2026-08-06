from pipelib.memory import measure_peak_inflight_microbatches


def test_peak_inflight_bounds():
    """Verify in-flight microbatches strictly respect theoretical pipeline bounds."""
    p = 4
    m = 16
    peaks = measure_peak_inflight_microbatches(p=p, m=m, schedule_type="1f1b")
    assert len(peaks) == p
    for stage, val in enumerate(peaks):
        expected_max = min(m, p - stage)
        assert val == expected_max, f"Stage {stage} peak {val} != expected {expected_max}"
