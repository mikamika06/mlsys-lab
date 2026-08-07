import numpy as np
import ref

def check(workdir):
    from checkpoint.dropout import verify_dropout_consistency
    out = {"dropout_matched": 0.0}
    x = np.ones((4, 4), dtype=float)
    p = 0.2
    _, mf1, mb1 = ref.simulate_dropout(x, p, 42, 42)
    _, mf2, mb2 = ref.simulate_dropout(x, p, 42, 99)

    res_good = verify_dropout_consistency(mf1, mb1)
    res_bad = verify_dropout_consistency(mf2, mb2)

    if res_good and not res_bad:
        out["dropout_matched"] = 1.0
    return out
