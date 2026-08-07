from serving.simulator import simulate_tiering

def test_host_receives_evictions():
    reqs = [
        {"id": 0, "arrive": 0, "seq": (1, 2, 3)},
        {"id": 1, "arrive": 1, "seq": (4, 5, 6)},
        {"id": 2, "arrive": 2, "seq": (1, 2)}
    ]
    gpu, host = simulate_tiering(reqs, gpu_c=3, host_c=3)
    assert host > 0.0, "Host cache did not serve hits from evicted GPU tokens."
