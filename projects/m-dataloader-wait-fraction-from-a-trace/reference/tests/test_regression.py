from pipeline.workers import min_workers_to_saturate


def test_workers_basic():
    assert min_workers_to_saturate(10.0, 2.0) == 5
    assert min_workers_to_saturate(1.0, 5.0) == 1
