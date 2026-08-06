import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from artifact_matrix.container_patch import resolve_container_patch

    out = {"patch_resolutions_matched": 1.0}

    for idx, (c_ver, e_ver, policy) in enumerate(ref.PATCH_INPUTS):
        want = ref.ref_resolve_container_patch(c_ver, e_ver, policy)
        got = resolve_container_patch(c_ver, e_ver, policy)
        if got != want:
            out["patch_resolutions_matched"] = 0.0
            out["_note"] = f"patch case {idx} ({c_ver}, {e_ver}, {policy}): got {got}, want {want}"
            break

    return out
