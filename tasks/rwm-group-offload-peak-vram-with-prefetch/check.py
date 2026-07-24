import numpy as np


def _oracle(group_sizes: np.ndarray, leaf_sizes: np.ndarray) -> dict:
    return {
        "group": float(2.0 * np.max(group_sizes)),
        "sequential": float(np.max(leaf_sizes)),
        "model": float(np.sum(group_sizes)),
    }


def _make_partition(rng):
    """Build a consistent (group_sizes, leaf_sizes) pair for the same model:
    generate leaves first, then cut them into contiguous groups so
    leaf_sizes.sum() == group_sizes.sum() exactly."""
    n_leaves = int(rng.integers(6, 40))
    leaf_sizes = rng.integers(1, 5_000_000, size=n_leaves).astype(np.float64)

    n_groups = int(rng.integers(2, min(8, n_leaves) + 1))
    # n_groups - 1 distinct cut points in [1, n_leaves - 1]
    cuts = sorted(rng.choice(np.arange(1, n_leaves), size=n_groups - 1, replace=False))
    bounds = [0] + list(cuts) + [n_leaves]
    group_sizes = np.array(
        [leaf_sizes[bounds[i]:bounds[i + 1]].sum() for i in range(len(bounds) - 1)],
        dtype=np.float64,
    )
    return group_sizes, leaf_sizes


def grade(sol, fx) -> dict:
    """
    Builds several random (group_sizes, leaf_sizes) partitions of the same
    model and checks the student's three reported peaks exactly match the
    NumPy-computed reference peaks on every trial.
    """
    rng = np.random.default_rng(0)
    n_trials = 6
    ok = 1.0

    for _ in range(n_trials):
        group_sizes, leaf_sizes = _make_partition(rng)
        expected = _oracle(group_sizes, leaf_sizes)

        try:
            got = sol.offload_peak_vram(group_sizes.copy(), leaf_sizes.copy())
        except Exception:
            ok = 0.0
            break

        try:
            if not isinstance(got, dict):
                ok = 0.0
                break
            for key in ("group", "sequential", "model"):
                if key not in got:
                    ok = 0.0
                    break
                if float(got[key]) != expected[key]:
                    ok = 0.0
                    break
        except Exception:
            ok = 0.0
            break

        if ok == 0.0:
            break

    return {"exact_match": ok}
