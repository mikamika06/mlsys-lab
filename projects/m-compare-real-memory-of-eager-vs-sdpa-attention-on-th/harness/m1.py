import ref
import torch


def check(workdir):
    from attnmem.model import TinyAttentionModel

    out = {"models_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        try:
            model_ref = ref.build_model(cfg)
            model_learner = TinyAttentionModel(cfg)
            x = torch.randn(1, 16, cfg["hidden_size"])
            model_ref.eval()
            model_learner.eval()
            with torch.no_grad():
                out_ref = model_ref(x, use_sdpa=False)
                out_learn = model_learner(x, use_sdpa=False)
            if out_ref.shape == out_learn.shape:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"config {i}: shape mismatch got {out_learn.shape}, want {out_ref.shape}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"config {i} raised {type(e).__name__}: {str(e)[:100]}"
    out["models_matched"] = float(ok)
    return out
