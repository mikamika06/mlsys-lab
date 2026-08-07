import sys
sys.path.insert(0, ".")
from engine.profile import identify_sensitive_layers
from engine.quantize import place_qdq_nodes

def test_identify_sensitive_layers_returns_correct_count():
    mses = [0.1, 0.5, 0.2, 0.9, 0.3, 0.4, 0.01, 0.05, 0.8, 0.15, 0.22, 0.11]
    sens = identify_sensitive_layers(mses, top_k=3)
    assert len(sens) == 3
    assert 3 in sens

def test_place_qdq_nodes_excludes_sensitive():
    graph = {"nodes": [{"name": f"layer_{i}"} for i in range(12)]}
    sensitive = [1, 3, 5]
    res = place_qdq_nodes(graph, sensitive)
    assert res["nodes"][1]["has_qdq"] is False
    assert res["nodes"][0]["has_qdq"] is True
