import sys
sys.path.insert(0, ".")
from heterogeneous.isolation import RunnerState, execute_request

def test_num_ctx_does_not_persist():
    runner = RunnerState()
    runner.num_ctx = 2048
    req1 = {"num_ctx": 4096, "prompt": "hello"}
    req2 = {"prompt": "world"}
    execute_request(runner, req1)
    assert runner.num_ctx == 2048, "num_ctx leaked past request boundary"
    execute_request(runner, req2)
    assert runner.num_ctx == 2048
