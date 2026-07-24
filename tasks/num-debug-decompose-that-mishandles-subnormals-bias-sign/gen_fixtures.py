"""No static fixtures.

The subnormal-heavy float32 vector is assembled inside `check.py` from raw
32-bit patterns with `np.random.default_rng(0)`, so grading is reproducible
without shipping binary files.
"""
