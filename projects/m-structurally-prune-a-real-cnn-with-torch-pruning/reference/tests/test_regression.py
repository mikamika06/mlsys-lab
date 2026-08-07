import sys
import torch
sys.path.insert(0, ".")
from cnnprune.toy import ToyNet, propagate_channels
from cnnprune.prune import SimpleCNN, structural_prune


def test_toy_propagation():
    res = propagate_channels([0, 2, 4])
    assert res["conv2_in"] == [0, 2, 4]


def test_structural_reduction():
    net = SimpleCNN()
    pruned = structural_prune(net, 0.5)
    assert pruned.conv1.out_channels == 8
    assert pruned.conv2.in_channels == 8
