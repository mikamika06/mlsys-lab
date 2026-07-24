def classify_descriptors(classes):
    result = []
    for cls in classes:
        result.append(
            hasattr(cls, "__set__") or hasattr(cls, "__delete__")
        )
    return result
