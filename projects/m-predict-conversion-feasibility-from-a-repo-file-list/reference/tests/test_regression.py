from ggufconv.feasibility import check_feasibility
from ggufconv.memory import estimate_conversion_memory
from ggufconv.tokenizer import compute_chkhsh


def test_missing_shard_detected_as_infeasible():
    files = [
        "config.json",
        "tokenizer.json",
        "model-00001-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    res = check_feasibility(files)
    assert res["feasible"] is False
    assert res["reason"] == "missing_shards"


def test_lazy_memory_is_less_than_eager_for_multishard():
    tensors = [
        {"name": "w1", "shape": [1024, 1024], "dtype": "float32", "shard_id": 1},
        {"name": "w2", "shape": [1024, 1024], "dtype": "float32", "shard_id": 1},
        {"name": "w3", "shape": [1024, 1024], "dtype": "float32", "shard_id": 2},
        {"name": "w4", "shape": [1024, 1024], "dtype": "float32", "shard_id": 2},
    ]
    lazy_res = estimate_conversion_memory(tensors, lazy=True)
    eager_res = estimate_conversion_memory(tensors, lazy=False)
    assert lazy_res["peak_memory_bytes"] < eager_res["peak_memory_bytes"]


def test_chkhsh_deterministic():
    h1 = compute_chkhsh(["hello", "world"], "llama-bpe")
    h2 = compute_chkhsh(["hello", "world"], "llama-bpe")
    assert h1 == h2
