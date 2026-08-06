import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    from lorameasure.params import count_trainable_params, sweep_ranks

    out = {"param_counts_matched": 0.0}
    model = ref.MODELS[0]
    targets = ["q_proj", "v_proj"]
    ranks = [2, 4, 8, 16]

    want_counts = ref.sweep_ranks(model, targets, ranks)
    got_counts = sweep_ranks(model, targets, ranks)

    if got_counts == want_counts:
        out["param_counts_matched"] = 1.0
    else:
        out["_note"] = f"Expected counts {want_counts}, got {got_counts}"

    return out
