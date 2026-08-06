def compute_relative_error(actual, predicted):
    if predicted == 0:
        return 0.0 if actual == 0 else 1.0
    return abs(actual - predicted) / float(predicted)
