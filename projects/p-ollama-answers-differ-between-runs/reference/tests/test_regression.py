import sys
sys.path.insert(0, ".")
from runner.client import ChatClient
from runner.config import merge_options

def test_options_priority():
    mf = {"seed": 1, "temperature": 0.7}
    api = {"temperature": 0.5}
    req = {"temperature": 0.0}
    res = merge_options(mf, api, req)
    assert res["temperature"] == 0.0
    assert res["seed"] == 1

def test_deterministic_generation():
    client = ChatClient()
    o1 = client.generate("test", seed=42, temperature=0.0)
    o2 = client.generate("test", seed=42, temperature=0.0)
    assert o1 == o2

def test_ten_runs_identical_hash():
    client = ChatClient()
    outputs = [client.generate("test", seed=42, temperature=0.0) for _ in range(10)]
    assert len(set(outputs)) == 1
