import sys
sys.path.insert(0, ".")
import torch
from dynamotrace.analyzer import analyze_function
from dynamotrace.rewrite import rewrite_fn

def test_graph_breaks_absent():
    def sample_fn(x):
        return x * 2.0

    optimized_fn = rewrite_fn(sample_fn)
    res = analyze_function(optimized_fn, (torch.randn(4),))
    assert res["break_count"] == 0, f"Expected 0 graph breaks, got {res['break_count']}"

def test_graph_count_minimal():
    def sample_fn(x):
        return x * 2.0

    optimized_fn = rewrite_fn(sample_fn)
    res = analyze_function(optimized_fn, (torch.randn(4),))
    assert res["graph_count"] == 1, f"Expected 1 graph count, got {res['graph_count']}"
