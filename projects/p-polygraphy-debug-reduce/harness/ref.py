import numpy as np
from poly.reducer import GraphReducer

def get_sample_data():
    return np.array([1.0, 2.0, 3.0])

def oracle_step_compare():
    r = GraphReducer([])
    ref_o = {"layer1": np.array([1.0, 2.0])}
    test_o = {"layer1": np.array([1.0, 2.5])}
    return r.step_compare(ref_o, test_o)

def oracle_bisection():
    nodes = [lambda x: x + 1, lambda x: x * 2, lambda x: x - 5]
    r = GraphReducer(nodes)
    oracle_fn = lambda ns, x: x + 1
    test_fn = lambda ns, x: x + 1 if len(ns) < 3 else (x + 1) * 2
    return r.bisection(get_sample_data(), oracle_fn, test_fn)

def oracle_isolate():
    r = GraphReducer([lambda x: x * 10])
    return r.isolate(0, np.array([2.0]))

def oracle_patch():
    r = GraphReducer([lambda x: x + 1])
    r.patch(0, lambda x: x + 2)
    return r.nodes[0](np.array([1.0]))

def oracle_verify():
    r = GraphReducer([lambda x: x + 5])
    return r.verify(np.array([1.0]), np.array([6.0]))
