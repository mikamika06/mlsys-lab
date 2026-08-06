import sys
import numpy as np
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from specdec.sampling import rejection_sample
    except Exception as e:
        return {"sampling_correct": 0.0, "dist_mse": 1.0, "_note": f"Import failed: {e}"}

    target_p, draft_p, p0, q0 = ref.get_m1_dist_fixture()

    rng = np.random.default_rng(12345)
    draft_tokens = np.array([1], dtype=np.int64)
    tokens, num_acc = rejection_sample(target_p, draft_p, draft_tokens, rng)

    if not isinstance(tokens, np.ndarray) or not isinstance(num_acc, (int, np.integer)):
        return {"sampling_correct": 0.0, "dist_mse": 1.0, "_note": "Incorrect return types"}

    if len(tokens) < 1 or len(tokens) > 2:
        return {"sampling_correct": 0.0, "dist_mse": 1.0, "_note": "Invalid tokens array length"}

    n_samples = 10000
    rng_mc = np.random.default_rng(42)

    first_tokens = []
    for _ in range(n_samples):
        dt = np.array([rng_mc.choice(len(q0), p=q0)], dtype=np.int64)
        toks, _ = rejection_sample(target_p, draft_p, dt, rng_mc)
        first_tokens.append(toks[0])

    counts = np.bincount(first_tokens, minlength=len(p0))
    empirical_dist = counts / float(n_samples)

    dist_mse = float(np.mean((empirical_dist - p0) ** 2))

    return {
        "sampling_correct": 1.0,
        "dist_mse": dist_mse,
    }
