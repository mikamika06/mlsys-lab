import sys
sys.path.insert(0, ".")
from gbreak.analyzer import collect_breaks, group_breaks
from gbreak.optimizer import count_graphs, check_equivalence

def test_graph_break_count():
    model = {}
    inputs = [1, 2, 3]
    g_count = count_graphs(model, inputs)
    assert g_count <= 2

def test_model_equivalence():
    model = {}
    inputs = [1, 2, 3]
    assert check_equivalence(model, model, inputs) is True
