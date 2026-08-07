import sys
sys.path.insert(0, ".")
from finetune.predictor import predict_fits
from finetune.fragmentation import analyze_memory_summary
from finetune.batchfinder import find_largest_batch_size

def test_predictor_basic():
    configs = [{"id": 1, "batch_size": 2, "seq_len": 512, "hidden_dim": 256, "num_layers": 4, "grad_accum": 1, "checkpointing": True}]
    res = predict_fits(configs, 10000000)
    assert len(res) == 1
    assert "fits" in res[0]

def test_fragmentation_basic():
    summary = "Allocated memory: 1000 bytes\nReserved memory: 1300 bytes"
    res = analyze_memory_summary(summary)
    assert "fragmentation_ratio" in res
    assert res["severity"] in ("low", "medium", "high")

def test_batchfinder_basic():
    curve = [{"batch_size": 1, "vram_bytes": 500}, {"batch_size": 2, "vram_bytes": 1200}, {"batch_size": 4, "vram_bytes": 3000}]
    assert find_largest_batch_size(curve, 1500) == 2
