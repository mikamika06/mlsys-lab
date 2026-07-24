def predict_class_reassignment(pairs):
    result = []
    for source, target in pairs:
        obj = source()
        try:
            obj.__class__ = target
            result.append(True)
        except TypeError:
            result.append(False)
    return result
