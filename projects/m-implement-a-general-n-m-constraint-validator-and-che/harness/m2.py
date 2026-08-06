import ref
import numpy as np


def check(workdir):
    from nmval.maskcheck import check_real_mask

    out = {"masks_matched": 0.0, "total": 5.0}
    ok = 0
    np.random.seed(123)
    for i in range(5):
        mask = np.random.choice([True, False], size=(16, 32), p=[0.5, 0.5])
        # Force validity on some
        if i % 2 == 0:
            flat = mask.reshape(-1, 4)
            for r in range(flat.shape[0]):
                idx = np.random.choice(4, size=2, replace=False)
                m_sub = np.zeros(4, dtype=bool)
                m_sub[idx] = True
                flat[r] = m_sub
        want_v, want_c = ref.check_mask(mask, 2, 4)
        got_v, got_c = check_real_mask(mask, 2, 4)
        if got_v == want_v and got_c == want_c:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"mask {i}: got ({got_v}), reference ({want_v})"
    out["masks_matched"] = float(ok)
    return out
