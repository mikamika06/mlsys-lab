def test_measure_breakdown_isolates_first_run():
    from ort_perf.profiler import measure_breakdown

    class MockSess:
        def __init__(self):
            pass
        def run(self, inputs):
            pass

    time_q = [0.0, 1.0, 11.0, 14.0]

    def time_fn():
        return time_q.pop(0)

    res = measure_breakdown(MockSess, None, time_fn, 3)
    assert res["first_run"] == 10.0
    assert res["steady_step"] == 1.0
