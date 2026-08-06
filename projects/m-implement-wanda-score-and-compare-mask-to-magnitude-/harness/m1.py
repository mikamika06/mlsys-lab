import sys
import numpy as np
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from pruning.wanda import magnitude_mask, wanda_mask, mask_recall
    except ImportError:
        return {"_note": "failed to import pruning.wanda"}

    rng = np.random.RandomState(42)
    W = rng.randn(16, 64)
    X = rng.randn(128, 64)

    out = {"mag_match": 0.0, "wan_match": 0.0, "rec_match": 0.0}
    try:
        m_mag_ref = ref.magnitude_mask(W, 0.25)
        m_mag_got = magnitude_mask(W, 0.25)
        if np.array_equal(m_mag_ref, m_mag_got):
            out["mag_match"] = 1.0

        m_wan_ref = ref.wanda_mask(W, X, 0.25)
        m_wan_got = wanda_mask(W, X, 0.25)
        if np.array_equal(m_wan_ref, m_wan_got):
            out["wan_match"] = 1.0

        rec_ref = ref.mask_recall(m_mag_ref, m_wan_ref)
        rec_got = mask_recall(m_mag_ref, m_wan_ref)
        if abs(rec_ref - rec_got) < 1e-5:
            out["rec_match"] = 1.0

    except NotImplementedError:
        pass

    return out
