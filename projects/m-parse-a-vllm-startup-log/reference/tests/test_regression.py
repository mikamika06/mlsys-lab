import sys
sys.path.insert(0, ".")
from vllmlog.parser import parse_log
from vllmlog.sharding import check_sharding
from vllmlog.diagnose import diagnose_garbage

def test_parser_basic():
    log = "INFO 08-12 10:00:00 llm_engine.py:73] Initializing an LLM engine (v0.6.0) with config: model='test', tensor_parallel_size=2, dtype=torch.float16, quantization=none"
    res = parse_log(log)
    assert res["tensor_parallel_size"] == 2
    assert res["model"] == "test"

def test_sharding_rule():
    assert check_sharding(32, 8, 2) is True
    assert check_sharding(32, 6, 4) is False

def test_diagnosis_tp_gt_1():
    res = diagnose_garbage(2, False, "none", "garbage")
    assert res == "invalid_sharding"
