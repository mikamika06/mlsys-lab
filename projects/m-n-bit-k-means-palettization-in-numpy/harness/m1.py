import ref
import numpy as np


def check(workdir):
    from palettize.kmeans import kmeans_palettize

    out = {"mse_matched": 0.0}
    matched = 0
    for t in ref.TENSORS:
        c_ref, l_ref = ref.kmeans_palettize(t, 3, vector_length=1)
        recon_ref = c_ref[l_ref].ravel()[:t.size]
        mse_ref = float(np.mean((t.ravel() - recon_ref) ** 2))

        try:
            c_got, l_got = kmeans_palettize(t, 3, vector_length=1)
            recon_got = c_got[l_got].ravel()[:t.size]
            mse_got = float(np.mean((t.ravel() - recon_got) ** 2))
            if np.isclose(mse_ref, mse_got, rtol=1e-2, atol=1e-2):
                matched += 1
        except Exception:
            pass

    out["mse_matched"] = float(matched)
    return out
