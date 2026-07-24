def classify_learning_rates(lrs, L):
    limit = 2.0 / float(L)
    return [int(lr >= limit) for lr in lrs]
