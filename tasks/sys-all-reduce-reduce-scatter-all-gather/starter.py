import numpy as np

def reduce_scatter(data, op="sum"):
    """Perform a reduce-scatter over n processes."""
    raise NotImplementedError("Implement reduce_scatter")

def all_gather(data):
    """Perform an all-gather over n processes."""
    raise NotImplementedError("Implement all_gather")
