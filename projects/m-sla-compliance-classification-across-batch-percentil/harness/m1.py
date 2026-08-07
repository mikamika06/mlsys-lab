import ref


def check(workdir):
    from sla.profiler import classify_sla_compliance

    out = {"profiles_matched": 0.0, "total_profiles": float(len(ref.PROFILES))}
    want = ref.reference_classify(ref.PROFILES, ref.TARGET_SLA)

    try:
        got = classify_sla_compliance(ref.PROFILES, ref.TARGET_SLA)
    except Exception as e:
        out["_note"] = f"classify_sla_compliance raised {type(e).__name__}: {e}"
        return out

    if not isinstance(got, dict) or "results" not in got or "max_compliant_batch" not in got:
        out["_note"] = "Return shape must be dict with keys 'results' and 'max_compliant_batch'"
        return out

    if got.get("max_compliant_batch") != want.get("max_compliant_batch"):
        out["_note"] = f"max_compliant_batch mismatch: got {got.get('max_compliant_batch')}, want {want.get('max_compliant_batch')}"
        return out

    matched = 0
    for b in ref.PROFILES:
        res_got = got["results"].get(b, {})
        res_want = want["results"][b]

        if res_got.get("compliant") == res_want["compliant"] and res_got.get("violations") == res_want["violations"]:
            pcts_ok = True
            for p, val in res_want["percentiles"].items():
                got_val = res_got.get("percentiles", {}).get(p)
                if got_val is None or abs(got_val - val) > 1e-4:
                    pcts_ok = False
                    break
            if pcts_ok:
                matched += 1

    out["profiles_matched"] = float(matched)
    return out
