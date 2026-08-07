import sys
sys.path.insert(0, ".")
from bench.harness import benchmark

def test_warmup_calls():
    calls = []
    def fn():
        calls.append(1)

    benchmark(fn, 15, 10, [50, 90])
    assert len(calls) == 25, f"Expected 25 calls, got {len(calls)}"

def test_no_times_handled():
    def fn():
        pass
    res = benchmark(fn, 0, 0, [50])
    assert res == {50: 0.0}
