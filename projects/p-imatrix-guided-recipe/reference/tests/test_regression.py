import sys
import numpy as np

sys.path.insert(0, ".")
from gguf_recipe.collector import collect_imatrix
from gguf_recipe.selector import select_quant_types
from gguf_recipe.quantizer import build_recipes
from gguf_recipe.evaluator import evaluate_recipe, measure_gain

def test_imatrix_non_zero():
    w = {"l": np.random.randn(8, 8)}
    c = np.random.randn(5, 8)
    im = collect_imatrix(w, c)
    assert np.all(im["l"] >= 0)

def test_selection_behavior():
    w = {"l": np.random.randn(8, 8)}
    im = {"l": np.ones((8, 8)) * 5.0}
    res = select_quant_types(w, im, threshold=1.0)
    assert res["l"] == "Q8_0"

def test_recipe_generation():
    w = {"l": np.random.randn(8, 8)}
    im = {"l": np.ones((8, 8))}
    rw, rwo = build_recipes(w, im)
    assert len(rw) == 1 and len(rwo) == 1

def test_evaluation_metrics():
    w = {"l": np.random.randn(8, 8)}
    r = {"l": "Q4_0"}
    res = evaluate_recipe(w, r)
    assert "ppl" in res and "kld" in res

def test_measure_gain_positive():
    w = {"l": np.random.randn(8, 8)}
    rw = {"l": "Q8_0"}
    rwo = {"l": "Q4_0"}
    gain = measure_gain(w, rw, rwo)
    assert gain > 0
