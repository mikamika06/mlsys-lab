import ref


def check(workdir):
    from imatrix_analysis.roles import rank_tensor_roles

    out = {"roles_ranked": 0.0}
    tensors_data = ref.generate_fixtures(seed=123)

    want = ref.rank_tensor_roles(tensors_data)
    try:
        got = rank_tensor_roles(tensors_data)
        if got == want:
            out["roles_ranked"] = 1.0
        else:
            out["_note"] = f"Expected ranking {want}, got {got}"
    except Exception as e:  # noqa: BLE001
        out["_note"] = f"Error evaluating rank_tensor_roles: {type(e).__name__}: {e}"

    return out
