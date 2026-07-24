import numpy as np

def compute_shared_e8m0_scale(weights):
    # TODO: this implementation incorrectly uses floor instead of ceil,
    # which produces a smaller exponent for values slightly above 6.
    amax = np.max(np.abs(weights), axis=
