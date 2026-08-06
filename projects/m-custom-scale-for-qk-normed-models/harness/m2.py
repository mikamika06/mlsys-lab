import numpy as np
import ref


def check(workdir):
    from qknorm.config import AttentionConfig
    from qknorm.attention import compute_qknorm_attention

    errs = []
    cases = [
        {"seed": 201, "scale": 0.2, "softcap": 5.0},
        {"seed": 202, "scale": None, "softcap": 10.0},
        {"seed": 203, "scale": 0.1, "softcap": 15.0},
    ]

    for c in cases:
        tc = ref.generate_test_case(seed=c["seed"], custom_scale=c["scale"], softcap=c["softcap"])
        config = AttentionConfig(
            head_dim=tc["head_dim"],
            custom_scale=tc["custom_scale"],
            softcap=tc["softcap"],
            eps=tc["eps"],
        )
        got = compute_qknorm_attention(tc["q"], tc["k"], tc["v"], config)
        expected = ref.oracle_qknorm_attention(
            tc["q"], tc["k"], tc["v"],
            head_dim=tc["head_dim"],
            custom_scale=tc["custom_scale"],
            softcap=tc["softcap"],
            eps=tc["eps"],
        )
        diff = np.linalg.norm(got - expected) / (np.linalg.norm(expected) + 1e-12)
        errs.append(diff)

    max_err = float(max(errs)) if errs else 1.0
    return {"rel_err": max_err}
