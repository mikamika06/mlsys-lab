import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from fa_classifier.dispatch import resolve_dispatch_target

    out = {"dispatch_matched": 0.0}
    ok = 0
    total = len(ref.TEST_PACKAGES) * len(ref.TEST_HARDWARE)

    for i, pkg in enumerate(ref.TEST_PACKAGES):
        for j, hw in enumerate(ref.TEST_HARDWARE):
            want = ref.resolve_dispatch_target(pkg, hw)
            try:
                got = resolve_dispatch_target(pkg, hw)
                if got == want:
                    ok += 1
                elif "_note" not in out:
                    out["_note"] = f"pair ({i}, {j}): got {got}, want {want}"
            except Exception as e:
                if "_note" not in out:
                    out["_note"] = f"pair ({i}, {j}) raised {type(e).__name__}: {e}"

    if ok == total:
        out["dispatch_matched"] = 1.0
    return out
