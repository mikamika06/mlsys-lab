import ref
import numpy as np


def check(workdir):
    from niaheval.harness import score_heatmap
    out = {"recall_at_k": 0.0}
    preds = {0.1: {100: [["apple", "banana"]]}}
    truths = {0.1: {100: ["apple"]}}
    got = score_heatmap(preds, truths, k=1)
    want = ref.score_heatmap(preds, truths, k=1)
    if np.allclose(got, want):
        out["recall_at_k"] = 1.0
    else:
        out["_note"] = f"heatmap mismatch: got {got}, want {want}"
    return out
