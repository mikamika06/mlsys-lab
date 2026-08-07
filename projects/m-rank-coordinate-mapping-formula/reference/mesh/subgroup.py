def get_subgroup_ranks(mesh_shape, fixed_axes):
    total = 1
    for d in mesh_shape:
        total *= d
    ranks = []
    for r in range(total):
        coords = __import__("mesh.mapping", fromlist=["rank_to_coords"]).rank_to_coords(r, mesh_shape)
        match = True
        for ax, val in fixed_axes.items():
            if coords[ax] != val:
                match = False
                break
        if match:
            ranks.append(r)
    return sorted(ranks)
