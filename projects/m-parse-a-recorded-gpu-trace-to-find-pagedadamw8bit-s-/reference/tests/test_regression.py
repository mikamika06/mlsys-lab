from parser.core import calculate_spillover_bytes, get_spike_index

def test_spike_index_correctness():
    assert get_spike_index([10, 10, 50, 20]) == 2
    assert get_spike_index([]) == -1

def test_spillover_bytes_respects_time_bounds():
    events = [
        {"name": "cudaMemcpyH2D", "ts": 10, "dur": 5, "args": {"bytes": 100}},
        {"name": "cudaMemcpyH2D", "ts": 30, "dur": 5, "args": {"bytes": 200}},
        {"name": "cudaMemcpyD2H", "ts": 50, "dur": 5, "args": {"bytes": 400}},
    ]
    res = calculate_spillover_bytes(events, 20, 40)
    assert res == 200
