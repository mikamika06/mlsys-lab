import numpy as np


def analyze_tp_compatibility(config):
    k = config["in_features"]
    n = config["out_features"]
    g = config["group_size"]
    desc_act = config.get("desc_act", False)
    tp_size = config["tp_size"]
    perm = config.get("perm", None)

    if k % tp_size != 0:
        return {
            "compatible": False,
            "reason": "k_not_divisible",
            "groups_per_rank": [],
            "is_contiguous_per_rank": [],
        }

    k_rank = k // tp_size
    if g > 0 and k_rank % g != 0 and not desc_act:
        return {
            "compatible": False,
            "reason": "group_not_divisible",
            "groups_per_rank": [],
            "is_contiguous_per_rank": [],
        }

    if perm is not None:
        perm_arr = np.array(perm, dtype=np.int32)
    else:
        perm_arr = np.arange(k, dtype=np.int32)

    g_idx = np.zeros(k, dtype=np.int32)
    if g > 0:
        for i in range(k):
            g_idx[perm_arr[i]] = i // g
    else:
        g_idx[:] = 0

    groups_per_rank = []
    is_contiguous_per_rank = []
    rank_group_sets = []

    for r in range(tp_size):
        slice_g = g_idx[r * k_rank : (r + 1) * k_rank]
        unique_g = np.unique(slice_g)
        groups_per_rank.append(int(len(unique_g)))
        rank_group_sets.append(set(unique_g.tolist()))

        if len(unique_g) == 0:
            is_contiguous_per_rank.append(True)
        else:
            min_g, max_g = int(np.min(unique_g)), int(np.max(unique_g))
            is_contig = (max_g - min_g + 1 == len(unique_g)) and np.array_equal(
                np.sort(unique_g), np.arange(min_g, max_g + 1)
            )
            is_contiguous_per_rank.append(bool(is_contig))

    overlap = False
    for r1 in range(tp_size):
        for r2 in range(r1 + 1, tp_size):
            if not rank_group_sets[r1].isdisjoint(rank_group_sets[r2]):
                overlap = True
                break

    all_contig = all(is_contiguous_per_rank)

    if desc_act and (not all_contig or overlap):
        compatible = False
        reason = "desc_act_group_misalignment"
    elif not all_contig or overlap:
        compatible = False
        reason = "group_misalignment"
    else:
        compatible = True
        reason = "ok"

    return {
        "compatible": compatible,
        "reason": reason,
        "groups_per_rank": groups_per_rank,
        "is_contiguous_per_rank": is_contiguous_per_rank,
    }
