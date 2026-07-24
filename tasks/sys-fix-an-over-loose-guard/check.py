import numpy as np


def _oracle_guard(cached_meta, new_meta):
    """Exact shape + dtype equality -- the only sound reuse condition."""
    return (cached_meta["shape"] == new_meta["shape"]
            and cached_meta["dtype"] == new_meta["dtype"])


def _op(x):
    """The traced computation: a trivial, deterministic elementwise op."""
    return np.asarray(x, dtype=np.float64) * 2.0 + 1.0


def _meta(x):
    return {"shape": tuple(x.shape), "dtype": str(x.dtype)}


def _run_cache(guard_fn, xs):
    """Drive a single-slot compiled-graph cache across a call sequence.

    If the guard says reuse is safe, the (stale) cached output is
    returned verbatim -- exactly what a real over-loose guard would do,
    since the whole point of reuse is to *skip* recomputation. Otherwise
    the op is retraced/recomputed fresh and becomes the new cache entry.
    """
    cache = None  # {"meta": ..., "y": ...}
    flags = []
    outputs = []
    for x in xs:
        meta = _meta(x)
        if cache is not None and guard_fn(cache["meta"], meta):
            flags.append(False)  # no recompile
            outputs.append(cache["y"])
        else:
            y = _op(x)
            cache = {"meta": meta, "y": y}
            flags.append(True)  # recompiled
            outputs.append(y)
    return flags, outputs


def _call_sequences():
    rng = np.random.default_rng(0)
    seqs = []

    # Same shape repeated, then a shape with equal numel but different
    # dims (must NOT reuse), then the exact original shape again (must
    # recompile since the cache slot now holds the (3,4) trace), then a
    # dtype-only change, then a genuinely unrelated shape.
    seqs.append([
        np.arange(12, dtype=np.float64).reshape(2, 6),
        np.arange(12, dtype=np.float64).reshape(2, 6) + 100.0,  # same shape/dtype -> reuse
        np.arange(12, dtype=np.float64).reshape(3, 4),          # same numel, diff shape
        np.arange(12, dtype=np.float64).reshape(2, 6) + 5.0,    # back to (2,6): must recompile
        np.arange(12, dtype=np.int64).reshape(2, 6),            # dtype change only
        np.arange(20, dtype=np.float64).reshape(4, 5),          # unrelated shape
    ])

    # Rank change with equal numel (1-D vs 2-D), and a 1x N vs N x 1 case.
    seqs.append([
        rng.standard_normal(8),
        rng.standard_normal((2, 4)),
        rng.standard_normal((2, 4)),   # repeat -> legit reuse
        rng.standard_normal((8, 1)),
        rng.standard_normal((1, 8)),
    ])

    # Larger random shapes, some repeated verbatim, some resized with
    # matching element counts.
    seqs.append([
        rng.standard_normal((3, 4, 2)),
        rng.standard_normal((3, 4, 2)),  # repeat -> legit reuse
        rng.standard_normal((4, 3, 2)),  # same numel (24), diff shape
        rng.standard_normal((24,)),      # same numel, rank change
        rng.standard_normal((6, 4)),     # same numel, diff shape
    ])

    return seqs


def grade(sol, fx) -> dict:
    for xs in _call_sequences():
        ref_flags, ref_outputs = _run_cache(_oracle_guard, xs)

        try:
            got_flags, got_outputs = _run_cache(sol.guard_ok, xs)
        except Exception:
            return {"exact_match": 0.0}

        if got_flags != ref_flags:
            return {"exact_match": 0.0}
        if len(got_outputs) != len(ref_outputs):
            return {"exact_match": 0.0}
        for g, r in zip(got_outputs, ref_outputs):
            g = np.asarray(g)
            if g.shape != r.shape or not np.array_equal(g, r):
                return {"exact_match": 0.0}

    return {"exact_match": 1.0}
