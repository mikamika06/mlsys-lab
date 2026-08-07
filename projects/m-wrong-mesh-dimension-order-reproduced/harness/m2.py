import numpy as np
import ref


def check(workdir):
    out = {"hsdp_verified": 0.0, "planner_matched": 0.0}

    try:
        from devmesh.hsdp import construct_hsdp_mesh, verify_hsdp_mesh
        from devmesh.planner import assign_fast_axis
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    try:
        u_mesh = construct_hsdp_mesh(world_size=16, replica_group_size=4, gpus_per_node=8)
        r_mesh = ref.construct_hsdp_mesh(world_size=16, replica_group_size=4, gpus_per_node=8)

        v1 = verify_hsdp_mesh(u_mesh, num_nodes=2, gpus_per_node=8)
        v2 = ref.verify_hsdp_mesh(r_mesh, num_nodes=2, gpus_per_node=8)

        inv_mesh = ref.DeviceMesh2D(
            mesh_shape=(4, 4),
            mesh_dim_names=("fsdp", "dp_replicate"),
            device_ids=np.arange(16).reshape((4, 4)),
        )
        inv_check = not verify_hsdp_mesh(inv_mesh, num_nodes=2, gpus_per_node=8)

        if v1 == v2 and inv_check and u_mesh.to_dict() == r_mesh.to_dict():
            out["hsdp_verified"] = 1.0
    except Exception as e:
        out["_note_hsdp"] = str(e)

    try:
        n = 8
        bw = np.ones((n, n), dtype=float) * 10.0
        for i in range(n):
            for j in range(n):
                if (i // 4) == (j // 4):
                    bw[i, j] = 900.0

        u_plan = assign_fast_axis(("dp", "tp"), (2, 4), bw, gpus_per_node=4)
        r_plan = ref.assign_fast_axis(("dp", "tp"), (2, 4), bw, gpus_per_node=4)

        if u_plan.to_dict() == r_plan.to_dict():
            out["planner_matched"] = 1.0
    except Exception as e:
        out["_note_planner"] = str(e)

    return out
