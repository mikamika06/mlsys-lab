import ref


def check(workdir):
    from mlx_vlm_utils.scaling import compute_token_count
    out = {"scaling_matches": 0.0}
    try:
        res = compute_token_count((512, 512), 16)
        want = ref.get_expected_scaling((512, 512), 16)
        if res == want:
            out["scaling_matches"] = 1.0
        else:
            out["_note"] = f"got token count {res}, want {want}"
    except Exception as e:
        out["_note"] = f"execution error: {type(e).__name__}: {str(e)[:120]}"
    return out
