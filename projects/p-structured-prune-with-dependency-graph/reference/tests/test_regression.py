import sys
sys.path.insert(0, ".")
import numpy as np
from prune.graph import DependencyGraph
from prune.group import GroupFinder
from prune.pruner import Pruner

def test_graph_nodes():
    g = DependencyGraph()
    g.add_node("layer1", (10, 10))
    assert "layer1" in g.nodes

def test_group_finder():
    g = DependencyGraph()
    g.add_node("l1", (10, 10))
    g.add_node("l2", (10, 10))
    g.add_edge("l1", "l2", 0, 1)
    gf = GroupFinder(g)
    groups = gf.find_groups()
    assert len(groups) == 1

def test_pruner_shapes():
    g = DependencyGraph()
    g.add_node("l1", (10, 10))
    model = {"l1": np.ones((10, 10))}
    p = Pruner(model, [["l1"]])
    p.prune_group(["l1"], [0])
    assert model["l1"].shape == (9, 10)
