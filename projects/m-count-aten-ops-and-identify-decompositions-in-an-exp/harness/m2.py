import ref


def check(workdir):
    from exportops.mutations import capture_export_mutation_error
    out = {"errors_captured": 0.0}
    ok = 0

    for m in ref.MODELS:
        got = capture_export_mutation_error(m, ref.dummy_export)
        want = None
        if m["mutates"]:
            want = ("ExportError", f"Unsupported global state mutation on '{m['mutates']}'")

        if got == want:
            ok += 1

    if ok == len(ref.MODELS):
        out["errors_captured"] = 1.0
    else:
        out["_note"] = f"matched {ok} of {len(ref.MODELS)}"

    return out
