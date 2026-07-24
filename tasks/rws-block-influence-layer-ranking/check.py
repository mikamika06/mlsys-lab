import numpy as np

def _oracle(x_in, x_out):
    # Compute block influence and ranking using the reference algorithm
    norms_in = np.linalg.norm(x_in, axis=2) + 1e-12
    norms_out = np.linalg.norm(x_out, axis=2) + 1e-12
    cos = (x_in * x_out).sum(axis=2) / (norms_in * norms_out)
    mean_cos = cos.mean(axis=0)
    influences = 1 - mean_cos
    ranking = list(np.argsort(-influences))
    return influences, ranking

def grade(sol, fx) -> dict:
    # Generate a handful of random test cases
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)
    cases = [
        (rng.integers(2,5), rng.integers(3,7), rng.integers(4,10)),
        (rng.integers(1,4), rng.integers(2,6), rng.integers(8,12)),
        (rng.integers(3,6), rng.integers(5,9), rng.integers(3,7))
    ]
    max_rel_err = 0.0
    all_rankings_match = True

    for batch, layers, feat in cases:
        x_in = rng.standard_normal((batch, layers, feat))
        x_out = rng.standard_normal((batch, layers, feat))

        try:
            influences, ranking = sol.block_influence_ranking(x_in, x_out)
        except Exception:
            return {"rel_err": 1e9, "spearman": 0.0}

        # Oracle
        oracle_inf, oracle_rank = _oracle(x_in, x_out)

        # Relative L2 error
        rel_err_case = np.linalg.norm(influences.astype(np.float64) - oracle_inf) / (
            np.linalg.norm(oracle_inf) + 1e-12)
        max_rel_err = max(max_rel_err, rel_err_case)

        # Ranking comparison
        if ranking != oracle_rank:
            all_rankings_match = False

    spearman_val = 1.0 if all_rankings_match else 0.0
    return {"rel_err": max_rel_err, "spearman": spearman_val}
