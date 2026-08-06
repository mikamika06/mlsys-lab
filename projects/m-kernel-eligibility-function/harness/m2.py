import sys
sys.path.insert(0, ".")
import ref


def check(workdir):
    out = {"labels_correct": 0.0, "remedies_correct": 0.0}
    try:
        from dispatch.analysis import label_checkpoints, resolve_minimal_remedies
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    ref_labels = ref.label_checkpoints(ref.KERNELS, ref.CHECKPOINTS)
    try:
        got_labels = label_checkpoints(ref.KERNELS, ref.CHECKPOINTS)
    except Exception as e:
        out["_note"] = f"label_checkpoints raised: {e}"
        return out

    if got_labels == ref_labels:
        out["labels_correct"] = 1.0
    else:
        out["_note"] = f"Labels mismatch: got {got_labels[:3]}, want {ref_labels[:3]}"
        return out

    ref_remedies = ref.resolve_minimal_remedies(ref.KERNELS, ref.CHECKPOINTS)
    try:
        got_remedies = resolve_minimal_remedies(ref.KERNELS, ref.CHECKPOINTS)
    except Exception as e:
        out["_note"] = f"resolve_minimal_remedies raised: {e}"
        return out

    if got_remedies == ref_remedies:
        out["remedies_correct"] = 1.0
    else:
        out["_note"] = f"Remedies mismatch: got {got_remedies[:2]}, want {ref_remedies[:2]}"

    return out
