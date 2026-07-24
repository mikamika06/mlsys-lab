import numpy as np

AMOUNT = 0.3


def _oracle(weights_list, amount):
    sizes = [w.size for w in weights_list]
    total = sum(sizes)
    flat_abs = np.concatenate([np.abs(w).ravel() for w in weights_list])
    k = int(round(amount * total))

    order = np.argsort(flat_abs, kind="stable")  # ascending magnitude
    prune_idx = order[:k]
    flat_mask = np.ones(total, dtype=bool)
    flat_mask[prune_idx] = False

    masks = []
    offset = 0
    for w in weights_list:
        n = w.size
        masks.append(flat_mask[offset:offset + n].reshape(w.shape))
        offset += n
    return masks, k


def _fail():
    return {"masks_exact_match": 0.0, "sparsity_off_by": float("inf")}


def grade(sol, fx) -> dict:
    weights = [fx["layer0"], fx["layer1"], fx["layer2"], fx["layer3"]]
    masks_ref, k_target = _oracle(weights, AMOUNT)

    try:
        got = sol.global_unstructured_masks([w.copy() for w in weights], AMOUNT)
    except Exception:
        return _fail()

    try:
        got = [np.asarray(m).astype(bool) for m in got]
    except Exception:
        return _fail()

    if len(got) != len(masks_ref):
        return _fail()
    for g, r in zip(got, masks_ref):
        if g.shape != r.shape:
            return _fail()

    masks_exact_match = 1.0 if all(np.array_equal(g, r) for g, r in zip(got, masks_ref)) else 0.0
    pruned_count = sum(int(np.sum(~g)) for g in got)
    sparsity_off_by = float(abs(pruned_count - k_target))

    return {"masks_exact_match": masks_exact_match, "sparsity_off_by": sparsity_off_by}
