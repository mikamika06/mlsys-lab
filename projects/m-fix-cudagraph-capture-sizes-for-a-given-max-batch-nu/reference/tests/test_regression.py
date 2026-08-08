import sys
sys.path.insert(0, ".")
from specengine.cudagraph import get_capture_sizes
from specengine.eagle import optimal_eagle_config
from specengine.parser import parse_trt_llm_log

def test_cudagraph_sizes_non_empty():
    sizes = get_capture_sizes(32, 4)
    assert len(sizes) > 0
    assert sizes[-1] == 32

def test_eagle_config_selection():
    cands = [
        {"draft_tokens": 3, "acceptance_rate": 0.5, "kv_per_token_bytes": 1024, "max_context_len": 1024, "max_batch_size": 4},
        {"draft_tokens": 5, "acceptance_rate": 0.6, "kv_per_token_bytes": 1024, "max_context_len": 1024, "max_batch_size": 4}
    ]
    cfg = optimal_eagle_config(100 * 1024 * 1024, cands)
    assert cfg["draft_tokens"] == 5

def test_parser_extraction():
    log = "DraftEngine initialized with layers=32 hidden=4096 spec_tokens=5\nPeak memory: 1024.5 MB"
    res = parse_trt_llm_log(log)
    assert res["layers"] == 32
    assert res["peak_memory_mb"] == 1024.5
