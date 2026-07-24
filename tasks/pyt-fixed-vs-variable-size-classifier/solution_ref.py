import sys
import numpy as np

def classify_objects(objs):
    res = []
    for obj in objs:
        t = type(obj)
        try:
            empty = t()
        except Exception:
            res.append(False)
            continue
        res.append(sys.getsizeof(obj) > sys.getsizeof(empty))
    return np.array(res, dtype=bool)
