from cbsim.engine import simulate_continuous, Request

def test_continuous_fills_batch():
    requests = [
        Request(1, 0, 10, 5),
        Request(2, 0, 10, 5),
        Request(3, 0, 10, 5),
    ]
    ticks, log = simulate_continuous(requests, 3)
    assert log[0] == 3
