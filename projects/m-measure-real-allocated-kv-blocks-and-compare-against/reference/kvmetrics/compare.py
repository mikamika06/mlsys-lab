def compare_allocation(predicted, actual):
    if actual == 0:
        return 0.0 if predicted == 0 else 1.0
    return abs(predicted - actual) / float(actual)
