from prof.model import compute_comm_bound_ratio

def test_comm_bound_ratio_ring_formula():
    timings = {"step_time": 0.100}
    params = {
        "msg_size_bytes": 100 * 1024 * 1024,
        "world_size": 8,
        "bandwidth_bytes_per_sec": 10 * 1024 * 1024 * 1024
    }
    ratio = compute_comm_bound_ratio(timings, params)
    expected = (2.0 * (7.0 / 8.0) * (100.0 / 10240.0) / 0.100) * 100.0
    assert abs(ratio - expected) < 1e-4
