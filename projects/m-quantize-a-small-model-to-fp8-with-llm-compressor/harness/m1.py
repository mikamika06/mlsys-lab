import ref


def check(workdir):
    from fp8quant.recipe import build_recipe
    out = {"recipe_matched": 0.0}
    try:
        got = build_recipe()
        want = ref.get_reference_recipe()
        if got == want:
            out["recipe_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)[:100]}"
    return out
