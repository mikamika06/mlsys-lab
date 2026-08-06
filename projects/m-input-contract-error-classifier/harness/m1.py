import ref

def check(workdir):
    try:
        from flash_contract.guard import check_contiguity
    except ImportError:
        return {"matches": 0.0, "_note": "failed to import check_contiguity"}

    ok = 0
    total = len(ref.STRIDES_TO_TEST)

    for s in ref.STRIDES_TO_TEST:
        try:
            got = check_contiguity(s)
            want = ref.check_contiguity(s)
            if got == want:
                ok += 1
        except Exception:
            pass

    return {"matches": 1.0 if ok == total else ok / float(total)}
