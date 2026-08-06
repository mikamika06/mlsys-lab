def derive_split_reorder_extents(original_extent, factor, reorder_indices):
    outer_extent = (original_extent + factor - 1) // factor
    inner_extent = factor
    extents = [outer_extent, inner_extent]
    reordered_extents = [extents[i] for i in reorder_indices]
    return reordered_extents
