import ref

def check(workdir):
    from export_fixer import symbolic
    out = {"symbolic_asserted": 0.0}
    try:
        model = ref.MODELS[0]
        res = symbolic.assert_symbolic_axes(model, "batch")
        if res is True:
            out["symbolic_asserted"] = 1.0
    except Exception as e:
        out["_note"] = str(e)
    return out
