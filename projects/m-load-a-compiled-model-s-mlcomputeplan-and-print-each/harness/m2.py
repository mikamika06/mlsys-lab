import ref


def check(workdir):
    from mlplan.analysis import find_ane_rejections
    from mlplan.plan import parse_compute_plan
    from mlplan.profile import routing_fractions

    out = {"fractions_match": 0.0, "rejections_match": 0.0}
    frac_ok = True
    rej_ok = True

    for i, p in enumerate(ref.PLANS):
        parsed = ref.parse_compute_plan(p)

        want_frac = ref.routing_fractions(parsed)
        got_frac = routing_fractions(parsed)

        for k in ["ANE", "GPU", "CPU"]:
            if abs(got_frac.get(k, -1.0) - want_frac[k]) > 1e-5:
                frac_ok = False
                if "_note" not in out:
                    out["_note"] = f"plan {i} fractions mismatch on {k}: got {got_frac}, want {want_frac}"
                break

        want_rej = ref.find_ane_rejections(parsed)
        got_rej = find_ane_rejections(parsed)
        if got_rej != want_rej:
            rej_ok = False
            if "_note" not in out:
                out["_note"] = f"plan {i} rejections mismatch: got {got_rej}, want {want_rej}"

    if frac_ok:
        out["fractions_match"] = 1.0
    if rej_ok:
        out["rejections_match"] = 1.0

    return out
