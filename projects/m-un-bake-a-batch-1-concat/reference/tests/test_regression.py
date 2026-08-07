import sys
sys.path.insert(0, ".")
from export_fixer.unbake import unbake_concat
from export_fixer.symbolic import assert_symbolic_axes
from export_fixer.histogram import op_histogram_diff

def test_unbake_modifies_nodes():
    g = {"node": [{"op": "Concat", "attribute": [{"name": "axis", "i": 0}]}]}
    res = unbake_concat(g)
    assert res["unbaked"] is True

def test_symbolic_axes_validation():
    model = {"input": [{"shape": ["batch", 3, 224, 224]}]}
    assert assert_symbolic_axes(model, "batch") is True

def test_histogram_diff_calculation():
    d = {"Add": 5, "Relu": 2}
    t = {"Add": 3, "Relu": 2}
    diff = op_histogram_diff(d, t)
    assert diff["Add"] == 2
    assert diff["Relu"] == 0
