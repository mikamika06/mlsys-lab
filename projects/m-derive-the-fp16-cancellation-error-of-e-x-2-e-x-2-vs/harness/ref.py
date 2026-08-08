import numpy as np

def get_variance_test_data():
    np.random.seed(123)
    # Mean of 10.0 and std 0.05 -> variance 0.0025.
    # E[x^2] ~ 100.0. 100 in float16 has precision steps of 1/1024 ~ 0.097,
    # which is larger than the variance itself, guaranteeing catastrophic cancellation.
    return np.random.normal(10.0, 0.05, 500).astype(np.float16)
