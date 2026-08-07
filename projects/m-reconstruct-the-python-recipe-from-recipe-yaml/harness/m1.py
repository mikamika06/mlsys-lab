import ref


def check(workdir):
    from reciperec.reconstruct import reconstruct_recipe

    out = {"recipes_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.RECIPES):
        want = ref.reconstruct_recipe(cfg)
        try:
            got = reconstruct_recipe(cfg)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"recipe {i} raised {type(e).__name__}: {str(e)[:100]}"
            continue

        norm_want = "\n".join([line.strip() for line in want.splitlines() if line.strip()])
        norm_got = "\n".join([line.strip() for line in (got or "").splitlines() if line.strip()])
        if norm_got == norm_want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"recipe {i} mismatch:\nwant: {norm_want[:120]}\ngot: {norm_got[:120]}"

    out["recipes_matched"] = float(ok)
    return out
