import numpy as np
import ref


def check(workdir):
    from qknorm.config import AttentionConfig
    from qknorm.attention import compute_qknorm_attention

    errs = []
    seeds = [101, 102, 103]
    scales = [0.125, 0.5, None]

    for seed, scale in zip(seeds, scales):
        tc = ref.generate_test_case(seed=seed, custom_scale=scale)
        config = AttentionConfig(
            head_dim=tc["head_dim"],
            custom_scale=tc["custom_scale"],
            eps=tc["eps"],
        )
        got = compute_qknorm_attention(tc["q"], tc["k"], tc["v"], config)
        expected = ref.oracle_qknorm_attention(
            tc["q"], tc["k"], tc["v"],
            head_dim=tc["head_dim"],
            custom_scale=tc["custom_scale"],
            eps=tc["eps"],
        )
        diff = np.linalg.norm(got - expected) / (np.linalg.norm(expected) + 1e-12)
        errs.append(diff)

    max_err = float(max(errs)) if errs else 1.0
    return {"rel_err": max_err}
