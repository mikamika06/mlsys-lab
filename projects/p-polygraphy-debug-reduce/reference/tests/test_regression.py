import sys
sys.path.insert(0, ".")
import numpy as np
from poly.reducer import GraphReducer

def test_reduced_repro():
    nodes = [lambda x: x * 2, lambda x: x + 1, lambda x: x * -1]
    reducer = GraphReducer(nodes)
    inputs = np.array([1.0, 2.0])
    oracle = lambda ns, x: x * 2 + 1
    test_runner = lambda ns, x: x * 2 + 1 if len(ns) < 3 else (x * 2 + 1) * -1
    faulty = reducer.bisection(inputs, oracle, test_runner)
    assert faulty == 2
    reducer.patch(2, lambda x: x * 1)
    assert reducer.verify(inputs, np.array([3.0, 5.0]))
