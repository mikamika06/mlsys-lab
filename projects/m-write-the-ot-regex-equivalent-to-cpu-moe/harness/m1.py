import ref


def check(workdir):
    from ot.translator import translate_cpu_moe

    out = {"translations_matched": 0.0}
    ok = 0
    for item in ref.CONFIGS:
        try:
            got = translate_cpu_moe(item["flag"])
        except Exception:
            got = ""
        if got == item["expected"]:
            ok += 1
    out["translations_matched"] = float(ok)
    return out
