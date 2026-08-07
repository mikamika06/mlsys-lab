import numpy as np
import ref

def check(workdir):
    m = {"equivalent": 0.0}
    b = ref.get_reference_balancer(4, 64)
    arr1 = np.ones((4, 4))
    arr2 = np.ones((4, 4)) + 1e-8
    if b.verify_equivalence(arr1, arr2):
        m["equivalent"] = 1.0
    return m
