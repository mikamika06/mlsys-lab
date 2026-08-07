import sys
sys.path.insert(0, ".")
from prune_eval.finetune import evaluate_sweep

def test_recovery_improves_accuracy():
    ratios = [0.2, 0.4, 0.6, 0.8]
    res = evaluate_sweep(ratios)
    unrec = res["unrecovered"]
    rec = res["recovered"]
    for u, r in zip(unrec, rec):
        assert r >= u, f"recovered accuracy {r} is not greater than or equal to unrecovered {u}"

def test_recovery_strictly_better_at_high_ratios():
    ratios = [0.5, 0.7, 0.9]
    res = evaluate_sweep(ratios)
    unrec = res["unrecovered"]
    rec = res["recovered"]
    assert rec[-1] > unrec[-1], "fine-tuning showed no improvement at high ratio"
