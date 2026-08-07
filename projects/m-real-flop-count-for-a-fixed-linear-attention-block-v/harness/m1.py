import torch
import ref

def check(workdir):
    from flopcalc.profiler import measure_flops
    out = {"flops_matched": 0.0, "rel_err": 1.0}

    model = ref.DummyAttentionBlock(hidden_dim=32, num_heads=2)
    x = torch.randn(1, 16, 32)

    want_flops = ref.measure_block_flops(model, x)
    try:
        got_flops = measure_flops(model, x)
    except Exception as e:
        out["_note"] = f"execution raised {type(e).__name__}: {str(e)[:100]}"
        return out

    if got_flops is None or got_flops <= 0:
        out["_note"] = f"invalid flop count returned: {got_flops}"
        return out

    rel_err = abs(got_flops - want_flops) / max(1.0, float(want_flops))
    out["rel_err"] = float(rel_err)
    if rel_err < 0.05:
        out["flops_matched"] = 1.0
    else:
        out["_note"] = f"got flops {got_flops}, expected ~{want_flops}, rel_err {rel_err:.4f}"
    return out
