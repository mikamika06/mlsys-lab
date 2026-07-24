"""No external fixture files -- check.py builds seeded random (W, X)
trials directly with np.random.default_rng(0), each with a couple of
outlier activation channels (the pattern Wanda's activation-aware
importance is designed to protect, and pure-magnitude pruning is
oblivious to)."""
