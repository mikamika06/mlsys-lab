import ref

EASY = ("01_literal", "02_trim_left", "03_trim_right", "04_range_index", "05_if_empty")


def check(workdir):
    from gotmpl import render

    picked = [c for c in ref.pick("semantics")
              if any(c[1]["template"].startswith(p) for p in EASY)]
    frac, ok, total = ref.score(render, picked)
    return {"basic_render": frac, "cases_passed": float(ok), "cases_total": float(total)}
