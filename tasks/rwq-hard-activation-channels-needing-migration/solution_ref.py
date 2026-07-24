import numpy as np

def migration_channels(X, threshold):
    """Return sorted list of channel indices whose absmax > threshold * median_absmax."""
    X = np.asarray(X, dtype=np.float64)
    channel_absmax = np.max(np.abs(X), axis=0)
    rho = np.median(channel_absmax)
    flagged = np.where(channel_absmax > threshold * rho)[0]
    return sorted(int(i) for i in flagged)
