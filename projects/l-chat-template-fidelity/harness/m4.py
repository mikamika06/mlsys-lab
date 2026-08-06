import ref


def check(workdir):
    from gotmpl import render

    picked = ref.pick("ollama", templates=["devstral"], with_tools=True)
    frac, ok, total = ref.score(render, picked)
    return {"devstral_tools": frac, "cases_passed": float(ok),
            "cases_total": float(total)}
