import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from fa_classifier.classifier import classify_package

    out = {"packages_matched": 0.0}
    ok = 0
    total = len(ref.TEST_PACKAGES)

    for i, pkg in enumerate(ref.TEST_PACKAGES):
        want = ref.classify_package(pkg)
        try:
            got = classify_package(pkg)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"package {i}: got {got}, want {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"package {i} raised {type(e).__name__}: {e}"

    if ok == total:
        out["packages_matched"] = 1.0
    return out
