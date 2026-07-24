def classify_iterators(objects):
    result = []
    for obj in objects:
        try:
            result.append(iter(obj) is obj)
        except TypeError:
            result.append(False)
    return result
