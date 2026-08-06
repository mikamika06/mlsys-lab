import ref


def check(workdir):
    from gotmpl import render

    picked = ref.pick("ollama", templates=["devstral"], with_tools=False)
    frac, ok, total = ref.score(render, picked)
    return {"devstral_plain": frac, "cases_passed": float(ok),
            "cases_total": float(total)}
