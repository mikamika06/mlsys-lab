import numpy as np

def classify_mask_2_4(mask: np.ndarray):
    """
    Return a tuple (group_validity, overall) indicating whether each group of four
    consecutive elements along the last dimension sums to exactly two.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 0:
        raise ValueError("mask must have at least one dimension")
    if mask.shape[-1] % 4 != 0:
        raise ValueError("last dimension must be divisible by 4")
    shape_prefix = mask.shape[:-1]
    last_dim = mask.shape[-1]
    num_groups = last_dim // 4
    overall = True
    def build_validity(shape_idx):
        nonlocal overall
        if len(shape_idx) == len(shape_prefix):
            row_validity = []
            for g in range(num_groups):
                s = 0
                s += int(mask[shape_idx + (g * 4 + 0,)])
                s += int(mask[shape_idx + (g * 4 + 1,)])
                s += int(mask[shape_idx + (g * 4 + 2,)])
                s += int(mask[shape_idx + (g * 4 + 3,)])
                valid = (s == 2)
                if not valid:
                    overall = False
                row_validity.append(valid)
            return row_validity
        else:
            dim_idx = len(shape_idx)
            sub_list = []
            for i in range(shape_prefix[dim_idx]):
                sub_list.append(build_validity(shape_idx + (i,)))
            return sub_list
    if len(shape_prefix) == 0:
        row_validity = []
        for g in range(num_groups):
            s = 0
            s += int(mask[g * 4 + 0])
            s += int(mask[g * 4 + 1])
            s += int(mask[g * 4 + 2])
            s += int(mask[g * 4 + 3])
            valid = (s == 2)
            if not valid:
                overall = False
            row_validity.append(valid)
        group_validity_list = row_validity
    else:
        group_validity_list = build_validity(())
    group_validity = np.array(group_validity_list, dtype=bool)
    return group_validity, overall
