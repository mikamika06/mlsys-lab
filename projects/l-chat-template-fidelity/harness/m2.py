import ref

HARD = ("06_map_order", "07_var_assign", "08_and_root", "09_pipeline", "10_len",
        "11_with", "12_toolcalls", "13_gt_not", "14_truthy_int")


def check(workdir):
    from gotmpl import render

    picked = [c for c in ref.pick("semantics")
              if any(c[1]["template"].startswith(p) for p in HARD)]
    frac, ok, total = ref.score(render, picked)
    out = {"semantics_render": frac, "cases_passed": float(ok),
           "cases_total": float(total)}
    for name, key in (("06_map_order", "map_sorted"), ("07_var_assign", "assignment"),
                      ("09_pipeline", "pipelines")):
        sub = [c for c in picked if c[1]["template"].startswith(name)]
        out[key] = ref.score(render, sub)[0]
    return out
