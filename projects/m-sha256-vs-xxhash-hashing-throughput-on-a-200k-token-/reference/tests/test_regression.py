import sys
sys.path.insert(0, ".")
from hashbench.hashing import hash_sha256, hash_xxhash, measure_throughput
from hashbench.drift import simulate_template_drift


def test_hash_consistency():
    data = b"hello prefix caching world" * 1000
    assert hash_sha256(data) != hash_xxhash(data)
    assert len(hash_sha256(data)) == 64
    assert len(hash_xxhash(data)) == 16


def test_throughput_ratio():
    data = b"token block data " * 50000
    res = measure_throughput(data, iterations=2)
    assert res["ratio"] > 1.0


def test_drift_simulation():
    prompt = "User: hello assistant"
    drifted = simulate_template_drift(prompt, "whitespace")
    assert drifted != prompt
    none_case = simulate_template_drift(prompt, "none")
    assert none_case == prompt
