import sys
sys.path.insert(0, ".")
from recomp.starvation import derive_max_num_batched_tokens
from recomp.log_parser import parse_preempted_requests

def test_parse_basic_logs():
    logs = [
        "INFO 01-01 00:00:00 scheduler.py:100] Starting iteration",
        "WARNING 01-01 00:00:01 scheduler.py:250] RECOMPUTE triggered for request_id=req_abc123 due to memory pressure, num_tokens=512",
        "WARNING 01-01 00:00:02 scheduler.py:250] RECOMPUTE triggered for request_id=req_xyz789 due to memory pressure, num_tokens=1024"
    ]
    res = parse_preempted_requests(logs)
    assert len(res) == 2
    assert res[0]["request_id"] == "req_abc123"
    assert res[0]["num_tokens"] == 512

def test_derive_token_limit_bounds():
    waiting = [{"prompt_tokens": 100}, {"prompt_tokens": 200}]
    running = [{"seq_len": 500}, {"seq_len": 500}]
    limit = derive_max_num_batched_tokens(waiting, running, 0.5)
    assert limit > 0
    assert isinstance(limit, int)
