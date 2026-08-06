import sys
sys.path.insert(0, ".")
from vllm_engine.resolver import resolve_args
from vllm_engine.parser import parse_argv
from vllm_engine.buckets import classify_arguments

def test_resolver_precedence():
    defaults = {"port": 8000, "tensor_parallel_size": 1}
    yaml_cfg = {"port": 9000, "tensor_parallel_size": 2}
    env_cfg = {"port": 7000}
    cli_cfg = {"port": 6000}
    res = resolve_args(defaults, yaml_cfg, env_cfg, cli_cfg)
    assert res["port"] == 6000
    assert res["tensor_parallel_size"] == 2

def test_parser_argv():
    argv = ["--port=8000", "--host", "localhost", "--enable-chunked-prefill"]
    parsed = parse_argv(argv)
    assert parsed["port"] == "8000"
    assert parsed["host"] == "localhost"
    assert parsed["enable-chunked-prefill"] is True

def test_buckets():
    names = ["gpu_memory_utilization", "max_num_seqs", "tensor_parallel_size", "seed"]
    b = classify_arguments(names)
    assert len(b["memory"]) == 1
    assert len(b["scheduling"]) == 1
    assert len(b["latency"]) == 1
    assert len(b["correctness"]) == 1
