"""Generate random eigenvalue spectra and slice fractions for additional testing."""
import json
import numpy as np

rng = np.random.default_rng(42)
fixtures = []
for _ in range(30):
    d = int(rng.integers(5, 80))
    raw = rng.exponential(scale=np.linspace(10, 0.1, d))
    ev = np.sort(raw)[::-1].tolist()
    s = round(float(rng.uniform(0.05, 1.0)), 4)
    fixtures.append({"eigenvalues": ev, "s": s})

with open("fixtures.json", "w") as fh:
    json.dump(fixtures, fh)
