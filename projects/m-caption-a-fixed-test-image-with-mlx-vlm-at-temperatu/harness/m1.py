import ref


def check(workdir):
    from mlx_vlm_utils.caption import generate_caption
    out = {"caption_matches": 0.0}
    try:
        res = generate_caption("dummy_model", "dummy_image", "describe")
        want = ref.get_expected_caption("dummy_model", "dummy_image", "describe")
        if res == want:
            out["caption_matches"] = 1.0
        else:
            out["_note"] = f"got caption {res}, want {want}"
    except Exception as e:
        out["_note"] = f"execution error: {type(e).__name__}: {str(e)[:120]}"
    return out
