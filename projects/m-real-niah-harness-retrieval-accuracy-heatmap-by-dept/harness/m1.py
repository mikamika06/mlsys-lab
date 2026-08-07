import ref


def check(workdir):
    from niaheval.generator import generate_task
    out = {"generators_matched": 0.0}
    t = generate_task(100, 0.5, "test_needle")
    want = ref.generate_task(100, 0.5, "test_needle")
    if t.get("tokens") == want["tokens"] and t.get("pos") == want["pos"]:
        out["generators_matched"] = 1.0
    else:
        out["_note"] = f"generator output mismatch: got {t}, want {want}"
    return out
