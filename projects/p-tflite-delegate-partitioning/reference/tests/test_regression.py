import sys
sys.path.insert(0, ".")
from edge.model import analyze_graph
from edge.partitioner import find_unsupported_ops

def test_unsupported_ops_non_empty():
    ops = find_unsupported_ops("dummy")
    assert len(ops) > 0

def test_graph_analysis_structure():
    stats = analyze_graph("dummy")
    assert stats["total_ops"] > 0
