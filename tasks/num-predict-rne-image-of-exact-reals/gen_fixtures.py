"""No external fixture files; check.py builds its deterministic case list
(np.random.default_rng(0) + random.Random(12345) + engineered edge cases)
in-process."""
