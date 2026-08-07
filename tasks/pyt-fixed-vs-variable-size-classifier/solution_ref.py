import sys
from typing import Any, Iterable

def classify_objects(objs: Iterable[Any]) -> list[bool]:
    res = []
    for obj in objs:
        t = type(obj)
        try:
            empty = t()
        except Exception:
            res.append(False)
            continue
        res.append(sys.getsizeof(obj) > sys.getsizeof(empty))
    return res
