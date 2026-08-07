def predict_backend(evaluated, rejections):
    for b in evaluated:
        if b not in rejections:
            return b
    return evaluated[-1] if evaluated else None
