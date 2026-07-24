import random
import numpy as np

def _reference(up_proj, down_proj, target_width):
    importance = np.abs(up_proj).sum(axis=1) + np.abs(down_proj).sum(axis=0)
    idx = np.argsort(-importance)[:target_width]
    idx_sorted = np.sort(idx)
    up_sliced = up_proj[idx_sorted, :]
    down_sliced = down_proj[:, idx_sorted]
    return list(idx_sorted), up_sliced, down_sliced

def grade(sol, fx) -> dict:
    ok = 1.0
    for _ in range(5):
        in_dim = random.randint(3, 10)
        hidden_dim = random.randint(4, 12)
        out_dim = random.randint(2, 8)
        target_width = random.randint(1, hidden_dim - 1)

        up_proj = np.random.randn(hidden_dim, out_dim).astype(np.float64)
        down_proj = np.random.randn(in_dim, hidden_dim).astype(np.float64)

        try:
            got = sol.prune_ffn_neurons(up_proj, down_proj, target_width)
            if len(got) != 3:
                ok = 0.0
                break
            indices, up_sliced, down_sliced = got

            ref_indices, ref_up, ref_down = _reference(up_proj, down_proj, target_width)

            if indices != ref_indices:
                ok = 0.0
                break
            if up_sliced.shape != (target_width, out_dim) or down_sliced.shape != (in_dim, target_width):
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break

    return {"exact_match": ok}
