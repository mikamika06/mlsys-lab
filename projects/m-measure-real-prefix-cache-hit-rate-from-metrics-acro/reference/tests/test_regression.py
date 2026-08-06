import sys
sys.path.insert(0, ".")
from apcmetric.parser import parse_metrics

def test_parse_metrics_valid():
    raw = """
# HELP vllm:gpu_cache_config_prefix_cache_hit_total Prefix cache hits
vllm:gpu_cache_config_prefix_cache_hit_total 150.0
vllm:prompt_tokens_total 1000.0
"""
    res = parse_metrics(raw)
    assert res.get("vllm:gpu_cache_config_prefix_cache_hit_total") == 150.0
    assert res.get("vllm:prompt_tokens_total") == 1000.0

def test_parse_metrics_empty():
    assert parse_metrics("") == {}

def test_parse_metrics_comments():
    raw = "# just a comment\n\n"
    assert parse_metrics(raw) == {}
