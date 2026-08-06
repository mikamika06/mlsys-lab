import numpy as np

BAD = {
    "max_abs_err": float("inf"),
    "lse_abs_err": float("inf"),
    "mass_abs_err": float("inf"),
}


def _make_states(q, keys, values, splits):
    states = []
    for start, end in splits:
        k = keys[start:end]
        v = values[start:end]
        logits = q @ k.T
        m = np.max(logits, axis=1)
        weights = np.exp(logits - m[:, None])
        l = np.sum(weights, axis=1)
        o = weights @ v
        states.append((m.astype(np.float64), l.astype(np.float64), o.astype(np.float64)))
    return states


def _oracle(q, keys, values, splits):
    logits = q @ keys.T
    g = np.max(logits, axis=1)
    p = np.exp(logits - g[:, None])
    denom = np.sum(p, axis=1)

    output = (p @ values) / denom[:, None]
    global_lse = g + np.log(denom)

    mass = np.stack([np.sum(p[:, start:end], axis=1) / denom for start, end in splits], axis=0)
    return output, global_lse, mass


def _cases():
    rng = np.random.default_rng(3)

    yield (
        np.array([[0.2, -0.4], [1.1, 0.3]], dtype=np.float64),
        np.array([[0.5, 0.2], [-0.3, 0.8], [1.0, -0.7]], dtype=np.float64),
        np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]], dtype=np.float64),
        [(0, 1), (1, 3)],
    )
    yield (
        np.arange(12, dtype=np.float64).reshape(3, 4) / 5.0,
        np.array(
            [
                [0.2, 0.1, 0.0, 0.4],
                [0.7, -0.2, 0.3, -0.5],
                [0.6, 0.9, -0.1, 0.2],
                [-0.3, 0.4, 0.8, 0.1],
                [0.5, -0.7, 0.2, 0.6],
            ],
            dtype=np.float64,
        ),
        np.arange(25, dtype=np.float64).reshape(5, 5) / 7.0,
        [(0, 1), (1, 3), (3, 5)],
    )

    q = rng.normal(size=(3, 8))
    keys = rng.normal(size=(9, 8))
    keys[0:3] *= 250.0
    keys[6:9] *= 80.0
    yield q, keys, rng.normal(size=(9, 4)), [(0, 3), (3, 6), (6, 9)]

    q = rng.normal(size=(4, 5))
    keys = rng.normal(size=(11, 5)) * 6.0
    yield q, keys, rng.normal(size=(11, 3)), [(0, 1), (1, 9), (9, 11)]


def grade(sol, fx) -> dict:
    worst_out = 0.0
    worst_lse = 0.0
    worst_mass = 0.0

    for q, k, v, splits in _cases():
        states = _make_states(q, k, v, splits)
        ref_out, ref_lse, ref_mass = _oracle(q, k, v, splits)

        try:
            got = sol.reconstruct_output(states)
        except Exception:
            return BAD

        try:
            out, lse, mass = got
        except (TypeError, ValueError):
            return BAD

        try:
            out = np.asarray(out, dtype=np.float64)
            lse = np.asarray(lse, dtype=np.float64)
            mass = np.asarray(mass, dtype=np.float64)
        except (TypeError, ValueError):
            return BAD

        if out.shape != ref_out.shape or lse.shape != ref_lse.shape or mass.shape != ref_mass.shape:
            return BAD
        if not (np.all(np.isfinite(out)) and np.all(np.isfinite(lse)) and np.all(np.isfinite(mass))):
            return BAD

        worst_out = max(worst_out, float(np.max(np.abs(out - ref_out))))
        worst_lse = max(worst_lse, float(np.max(np.abs(lse - ref_lse))))
        worst_mass = max(worst_mass, float(np.max(np.abs(mass - ref_mass))))

    return {
        "max_abs_err": worst_out,
        "lse_abs_err": worst_lse,
        "mass_abs_err": worst_mass,
    }
