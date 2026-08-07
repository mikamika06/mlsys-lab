import ref
import torch

def check(workdir):
    from backends.counter import count_fx_nodes
    out = {"frequencies_matched": 0.0}
    models = ref.get_models()
    x = torch.randn(2, 16)
    matched = 0
    for i, m in enumerate(models):
        want = ref.compute_op_frequencies(m, x)
        try:
            got = count_fx_nodes(m, x)
            if got == want:
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"model {i}: got {got}, want {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"model {i} error: {str(e)[:100]}"
    out["frequencies_matched"] = float(matched)
    return out
