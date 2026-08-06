import ref
import numpy as np

def check(workdir):
    from ring.allreduce import ring_all_reduce

    np.random.seed(42)
    out = {"tensors_matched": 0.0, "total": 5.0}
    ok = 0

    for i in range(5):
        world_size = 4
        tensors = [np.random.randn(16).astype(np.float32) for _ in range(world_size)]
        want = ref.simulate_ring_allreduce(list(range(world_size)), tensors)
        got = ring_all_reduce(tensors)

        if got is not None and len(got) == world_size:
            match = True
            for g_t, w_t in zip(got, want):
                if not np.allclose(g_t, w_t, atol=1e-5):
                    match = False
                    break
            if match:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"tensor test {i} mismatch: got {got[0][:4]}, want {want[0][:4]}"
        elif "_note" not in out:
            out["_note"] = f"tensor test {i} returned invalid structure"

    out["tensors_matched"] = float(ok)
    return out
