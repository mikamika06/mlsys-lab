import ref


def check(workdir):
    from mesh.subgroup import get_subgroup_ranks

    out = {"subgroup_match": 0.0}
    ok = True
    for shape in ref.MESH_SHAPES:
        fixed_axes = {0: 0}
        got = get_subgroup_ranks(shape, fixed_axes)
        want = ref.get_subgroup_ranks(shape, fixed_axes)
        if got != want:
            ok = False
    out["subgroup_match"] = 1.0 if ok else 0.0
    return out
