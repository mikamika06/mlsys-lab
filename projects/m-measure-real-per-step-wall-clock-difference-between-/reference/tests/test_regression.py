import sys
sys.path.insert(0, ".")
from efficiency.bench import measure_step_latencies
from efficiency.memory import measure_memory_footprints, rank_memory_usage


def test_latency_ratio_bounds():
    cfg = {"steps_ft": [0.1, 0.1, 0.1], "steps_lora": [0.05, 0.05, 0.05]}
    res = measure_step_latencies(cfg)
    assert res["latency_ratio"] > 1.0


def test_memory_ranking_strict():
    cfg = {"base_memory": 500.0}
    mems = measure_memory_footprints(cfg)
    ranking = rank_memory_usage(mems)
    assert ranking == ["lora_4bit", "lora_bf16", "full_ft"]
