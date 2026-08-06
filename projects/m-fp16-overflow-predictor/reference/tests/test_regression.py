import sys
sys.path.insert(0, ".")
from fp16pred.predictor import predict_overflow
from fp16pred.ranking import rank_sensitivity, generate_golden

CONFIGS = [
    {"name": "layer_block_a", "scale": 12.5, "max_val": 40000.0, "mean": 1.2},
    {"name": "layer_block_b", "scale": 1.2, "max_val": 800.0, "mean": 0.05},
    {"name": "layer_block_c", "scale": 45.0, "max_val": 120000.0, "mean": 3.4},
]

def test_overflow_prediction_detects_extremes():
    stat = {"name": "test", "scale": 10.0, "max_val": 10000.0}
    assert predict_overflow(stat, threshold=65504.0) is True

def test_rank_sensitivity_ordering():
    ranked = rank_sensitivity(CONFIGS)
    assert len(ranked) == len(CONFIGS)
    assert ranked[0] == "layer_block_c"

def test_golden_file_keys():
    golden = generate_golden(CONFIGS)
    for cfg in CONFIGS:
        assert cfg["name"] in golden
