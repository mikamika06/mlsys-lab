import os
import tempfile
from profiler_util.instrument import run_training_loop, diagnose_double_step

def test_instrument_generates_traces():
    with tempfile.TemporaryDirectory() as tmpdir:
        params = {"wait": 1, "warmup": 1, "active": 2, "repeat": 1}
        count = run_training_loop(6, params, tmpdir)
        assert count == 1
        files = os.listdir(tmpdir)
        assert len(files) == 1

def test_diagnose_double_step():
    params = {"wait": 0, "warmup": 0, "active": 2, "repeat": 1}
    res = diagnose_double_step(3, params, 4)
    assert res == "double_step_detected"
