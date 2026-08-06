import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)

    from opt.grouping import group_tensors_by_device_dtype, estimate_kernel_counts

    out = {
        "groups_matched": 0.0,
        "kernel_counts_matched": 0.0
    }

    params = ref.generate_synthetic_params()

    try:
        user_groups = group_tensors_by_device_dtype(params)
        ref_groups = ref.reference_grouping(params)

        matched = True
        if set(user_groups.keys()) != set(ref_groups.keys()):
            matched = False
        else:
            for k in ref_groups:
                user_ids = sorted([p["id"] for p in user_groups[k]])
                ref_ids = sorted([p["id"] for p in ref_groups[k]])
                if user_ids != ref_ids:
                    matched = False
                    break
        if matched:
            out["groups_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"group_tensors_by_device_dtype failed: {e}"
        return out

    try:
        user_counts = estimate_kernel_counts(params, num_steps=5)
        ref_counts = ref.reference_kernel_counts(params, num_steps=5)

        if user_counts == ref_counts:
            out["kernel_counts_matched"] = 1.0
        else:
            out["_note"] = f"Kernel counts mismatch: got {user_counts}, expected {ref_counts}"
    except Exception as e:
        out["_note"] = f"estimate_kernel_counts failed: {e}"

    return out
