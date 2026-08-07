def score_magnitude(w):
    """
    Returns the magnitude of the weights.
    w: numpy array of shape (out_features, in_features)
    """
    raise NotImplementedError


def score_wanda(w, x):
    """
    Returns the Wanda importance score: |w| * ||x_j||_2
    w: numpy array of shape (out_features, in_features)
    x: calibration activations of shape (in_features, batch_size)
    """
    raise NotImplementedError
