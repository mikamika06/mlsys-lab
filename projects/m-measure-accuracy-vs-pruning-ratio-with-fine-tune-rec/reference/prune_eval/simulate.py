def measure_accuracy(model_mock, ratio, recovered=False):
    if recovered:
        return max(0.0, 1.0 - 0.2 * (ratio ** 2.0))
    else:
        return max(0.0, 1.0 - 0.5 * (ratio ** 1.5))
