import sys

sys.path.insert(0, ".")
from vllm_runner.config import build_command
from vllm_runner.client import parse_health, parse_completion


def test_build_command_contains_model():
    cfg = {"model": "facebook/opt-125m", "port": 8000, "tensor_parallel": 1, "max_model_len": 2048}
    cmd = build_command(cfg)
    assert "facebook/opt-125m" in cmd
    assert "docker" in cmd


def test_parse_health_valid():
    res = parse_health(200, "OK")
    assert res["ready"] is True


def test_parse_completion_extracts_text():
    payload = {"choices": [{"text": "generated output text"}]}
    text = parse_completion(payload)
    assert text == "generated output text"
