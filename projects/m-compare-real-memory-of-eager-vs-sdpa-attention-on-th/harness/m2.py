import ref
import torch


def check(workdir):
    from attnmem.model import TinyAttentionModel
    from attnmem.measure import compute_size_ratio

    out = {"ratio_match": 0.0, "sdpa_saves_memory": 0.0}
    try:
        cfg = ref.CONFIGS[0]
        model = TinyAttentionModel(cfg)
        x = torch.randn(2, 64, cfg["hidden_size"])
        learner_ratio = compute_size_ratio(model, x)
        ref_model = ref.build_model(cfg)
        ref_ratio = ref.compute_ref_ratio(ref_model, x)

        if abs(learner_ratio - ref_ratio) / max(1.0, ref_ratio) < 0.25:
            out["ratio_match"] = 1.0
        else:
            out["_note"] = f"ratio mismatch: got {learner_ratio}, reference ~{ref_ratio}"

        if learner_ratio > 1.0:
            out["sdpa_saves_memory"] = 1.0
        else:
            out["_note"] = f"expected SDPA to use less memory, got ratio {learner_ratio}"
    except Exception as e:
        out["_note"] = f"exception in m2: {type(e).__name__}: {str(e)[:100]}"
    return out
