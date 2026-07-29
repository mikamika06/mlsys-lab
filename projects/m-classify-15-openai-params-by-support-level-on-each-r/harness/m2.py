import ref


def check(workdir):
    from oaicompat import native_counters, shim_counters, hidden_counters

    out = {"native_match": 0.0, "shim_match": 0.0, "hidden_match": 0.0, "subset_invariant": 0.0}

    native_ok = True
    shim_ok = True
    hidden_ok = True
    subset_ok = True

    for runner in ref.RUNNERS:
        want_native = ref.native_counters(runner)
        got_native = list(native_counters(runner) or [])
        if got_native != want_native:
            native_ok = False
            if "_note" not in out:
                out["_note"] = f"native_counters({runner['name']}): got {got_native}, reference {want_native}"

        want_shim = ref.shim_counters(runner)
        got_shim = list(shim_counters(runner) or [])
        if got_shim != want_shim:
            shim_ok = False
            if "_note" not in out:
                out["_note"] = f"shim_counters({runner['name']}): got {got_shim}, reference {want_shim}"

        want_hidden = ref.hidden_counters(runner)
        got_hidden = list(hidden_counters(runner) or [])
        if got_hidden != want_hidden:
            hidden_ok = False
            if "_note" not in out:
                out["_note"] = f"hidden_counters({runner['name']}): got {got_hidden}, reference {want_hidden}"

        if not set(got_shim).issubset(set(got_native)):
            subset_ok = False
            if "_note" not in out:
                out["_note"] = f"shim_counters({runner['name']}) is not a subset of native_counters"

    out["native_match"] = 1.0 if native_ok else 0.0
    out["shim_match"] = 1.0 if shim_ok else 0.0
    out["hidden_match"] = 1.0 if hidden_ok else 0.0
    out["subset_invariant"] = 1.0 if subset_ok else 0.0
    return out
