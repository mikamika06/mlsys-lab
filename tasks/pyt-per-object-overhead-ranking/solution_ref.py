import sys


def rank_object_types():
    objects = [
        0,
        0.0,
        tuple(),
        "x",
        b"x",
        True,
        None,
        0j,
    ]
    keys = []
    for obj in objects:
        keys.append((sys.getsizeof(obj), type(obj).__name__))
    
    n = len(objects)
    indices = list(range(n))
    for i in range(n):
        for j in range(0, n - i - 1):
            if keys[indices[j]] > keys[indices[j + 1]]:
                indices[j], indices[j + 1] = indices[j + 1], indices[j]
                
    sorted_objects = [objects[i] for i in indices]
    return tuple(type(obj).__name__ for obj in sorted_objects)
