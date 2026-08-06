from benchcomp.ratios import compute_pairwise_ratios, rank_frameworks
from benchcomp.scaling import compute_scaling_efficiency


def test_pairwise_ratios():
    records = [
        {"framework": "A", "config_id": "c1", "num_gpus": 1, "tokens_per_sec": 100.0, "vram_gb": 10.0},
        {"framework": "B", "config_id": "c1", "num_gpus": 1, "tokens_per_sec": 50.0, "vram_gb": 15.0},
    ]
    ratios = compute_pairwise_ratios(records)
    assert abs(ratios[("A", "B")] - 2.0) < 1e-5
    assert abs(ratios[("B", "A")] - 0.5) < 1e-5


def test_rank_frameworks_vram_ascending():
    records = [
        {"framework": "A", "config_id": "c1", "num_gpus": 1, "tokens_per_sec": 100.0, "vram_gb": 20.0},
        {"framework": "B", "config_id": "c1", "num_gpus": 1, "tokens_per_sec": 200.0, "vram_gb": 10.0},
    ]
    ranking = rank_frameworks(records)
    assert ranking["speed"] == ["B", "A"]
    assert ranking["vram"] == ["B", "A"]


def test_scaling_efficiency():
    records = [
        {"framework": "A", "config_id": "c1", "num_gpus": 1, "tokens_per_sec": 100.0, "vram_gb": 10.0},
        {"framework": "A", "config_id": "c1", "num_gpus": 2, "tokens_per_sec": 180.0, "vram_gb": 11.0},
    ]
    eff = compute_scaling_efficiency(records)
    assert abs(eff[("A", 2)] - 0.90) < 1e-5
