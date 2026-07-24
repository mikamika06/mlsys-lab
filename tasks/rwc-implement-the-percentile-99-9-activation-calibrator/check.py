import numpy as np


def _quant_error(x, amax):
    scale = amax / 127.0
    q = np.clip(np.round(x / scale), -127, 127) * scale
    return float(np.mean((q - x) ** 2))


def _oracle_amax(batches):
    x = np.concatenate([np.asarray(b).reshape(-1) for b in batches])
    return float(np.percentile(np.abs(x), 99.9))


def grade(sol, fx) -> dict:
    cases = [
        [
            np.linspace(-3, 3, 1000, dtype=np.float64),
            np.array([50.0, -70.0, 90.0]),
        ],
        [
            np.random.default_rng(0).normal(0, 1, size=(4, 128)),
            np.array([25.0, -40.0]),
        ],
        [
            np.concatenate(
                [
                    np.random.default_rng(1).normal(0, 2, size=5000),
                    np.array([1000.0]),
                ]
            )
        ],
    ]

    max_rel = 0.0
    max_qerr = 0.0

    for batches in cases:
        try:
            got = float(sol.percentile_amax([np.array(b) for b in batches]))
        except Exception:
            return {"rel_err": 1.0, "quant_mse": 1.0}

        ref = _oracle_amax(batches)
        rel = abs(got - ref) / (abs(ref) + 1e-12)
        max_rel = max(max_rel, rel)

        x = np.concatenate([np.asarray(b).reshape(-1) for b in batches])
        q_ref = _quant_error(x, ref)
        q_got = _quant_error(x, got)
        max_qerr = max(max_qerr, abs(q_got - q_ref))

    return {
        "rel_err": float(max_rel),
        "quant_mse": float(max_qerr),
    }
