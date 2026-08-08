import ref
import numpy as np


def check(workdir):
    from palettize.compress import palettize_vector, palettize_size_bytes

    out = {"vector_mse_match": 0.0, "vector_size_match": 0.0}
    t = ref.generate_tensor()

    try:
        ref_p, ref_i = ref.palettize_vector(t, bits=8, block_size=2, iters=10)
        ref_recon = ref_p[ref_i].reshape(t.shape)

        p, i = palettize_vector(t, bits=8, block_size=2, iters=10)
        recon = p[i].reshape(t.shape)

        ref_mse = np.mean((t - ref_recon) ** 2)
        got_mse = np.mean((t - recon) ** 2)

        if abs(got_mse - ref_mse) < 1e-3:
            out["vector_mse_match"] = 1.0
        else:
            out["_note"] = f"vector mse mismatch: got {got_mse:.5f}, ref {ref_mse:.5f}"
    except Exception as e:
        out["_note"] = f"palettize_vector failed: {e}"
        return out

    try:
        sizes_ok = True
        for args in [(4096, 8, 2), (1024, 4, 4), (8192, 6, 2)]:
            if palettize_size_bytes(*args) != ref.palettize_size_bytes(*args):
                sizes_ok = False
        if sizes_ok:
            out["vector_size_match"] = 1.0
    except Exception as e:
        if "_note" not in out:
            out["_note"] = f"palettize_size_bytes failed: {e}"

    return out
