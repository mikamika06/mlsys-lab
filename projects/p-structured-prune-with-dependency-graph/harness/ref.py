import numpy as np
from prune.graph import DependencyGraph
from prune.group import GroupFinder
from prune.pruner import Pruner
from prune.eval import evaluate

def get_reference_model():
    return {"layer1": np.ones((16, 16)), "layer2": np.ones((16, 16))}
