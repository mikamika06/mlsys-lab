import ref

def check(workdir):
    from hpa.affinity import quantify_hit_rate_loss

    out = {"loss_match": 0.0}

    ok = 0
    total_checks = len(ref.SESSIONS) * 3
    try:
        for s in ref.SESSIONS:
            for rep in [2, 5, 10]:
                got = quantify_hit_rate_loss(s, rep)
                want = ref.quantify_hit_rate_loss(s, rep)
                if abs(got - want) < 1e-6:
                    ok += 1

        if ok == total_checks:
            out["loss_match"] = 1.0
        else:
            out["_note"] = f"quantify_hit_rate_loss failed some cases ({ok}/{total_checks} passed)"
    except Exception as e:
        out["_note"] = str(e)

    return out
