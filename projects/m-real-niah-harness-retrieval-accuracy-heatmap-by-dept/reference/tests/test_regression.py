import sys
sys.path.insert(0, ".")
from niaheval.generator import generate_task
from niaheval.harness import score_heatmap


def test_generator_bounds():
    t = generate_task(100, 0.5, "apple")
    assert len(t["tokens"]) == 100
    assert t["tokens"][t["pos"]] == "apple"


def test_heatmap_shape():
    preds = {0.1: {100: [["apple"]]}}
    truths = {0.1: {100: ["apple"]}}
    m = score_heatmap(preds, truths, k=1)
    assert m.shape == (1, 1)
    assert m[0, 0] == 1.0
