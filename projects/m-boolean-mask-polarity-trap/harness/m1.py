import torch
import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    out = {"exact_match": 0.0}
    try:
        import sdpa_trap.attention as att
        q, k, v, pad_mask = ref.get_fixture()
        expected = ref.get_expected()
        got = att.compute_manual(q, k, v, pad_mask)
        if torch.allclose(got, expected, atol=1e-5):
            out["exact_match"] = 1.0
        else:
            out["_note"] = "outputs do not match expected manual attention"
    except Exception as e:
        out["_note"] = f"error: {e}"
    finally:
        sys.path.pop(0)
    return out
