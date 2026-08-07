import ref


def check(workdir):
    from mesh.mapping import rank_to_coords, coords_to_rank

    out = {"mapping_match": 0.0}
    ok = True
    for shape in ref.MESH_SHAPES:
        total = 1
        for d in shape:
            total *= d
        for r in range(total):
            c = rank_to_coords(r, shape)
            if c != ref.rank_to_coords(r, shape):
                ok = False
            r2 = coords_to_rank(c, shape)
            if r2 != r:
                ok = False
    out["mapping_match"] = 1.0 if ok else 0.0
    return out
