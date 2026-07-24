"""Generate the arrivals fixture as a sorted numpy array of timestamps."""
import numpy as np

rng = np.random.default_rng(42)
intervals = rng.exponential(scale=0.5, size=1200)
arrivals = np.cumsum(intervals)
arrivals = arrivals[arrivals <= 500][:1000]
np.save('arrivals.npy', arrivals)
