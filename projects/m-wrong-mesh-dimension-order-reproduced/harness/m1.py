import ref


def check(workdir):
    try:
        from devmesh.mesh import DeviceMesh2D
    except Exception as e:
        return {"meshes_matched": 0.0, "_note": f"Import failed: {e}"}

    matched = 0.0
    for tc in ref.TEST_CASES:
        try:
            user_mesh = DeviceMesh2D(tc["shape"], tc["names"], tc["ranks"])
            ref_mesh = ref.DeviceMesh2D(tc["shape"], tc["names"], tc["ranks"])

            if user_mesh.to_dict() != ref_mesh.to_dict():
                continue

            c_ok = True
            for r in tc["ranks"]:
                if user_mesh.get_rank_coords(r) != ref_mesh.get_rank_coords(r):
                    c_ok = False
                    break
                for dim in tc["names"]:
                    if user_mesh.get_dim_group_ranks(r, dim) != ref_mesh.get_dim_group_ranks(r, dim):
                        c_ok = False
                        break
                    if user_mesh.is_fast_axis(dim) != ref_mesh.is_fast_axis(dim):
                        c_ok = False
                        break
            if c_ok:
                matched += 1.0
        except Exception:
            pass

    return {"meshes_matched": matched}
