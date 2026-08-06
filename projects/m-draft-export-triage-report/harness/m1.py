import ref


def check(workdir):
    from triage.exporter import run_draft_export

    out = {"modules_categorized": 0.0}
    ok = 0
    for model in ref.TEST_MODELS:
        want = ref.run_draft_export(model)
        got = run_draft_export(model)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"model {model['name']}: got {got}, reference {want}"

    if ok == len(ref.TEST_MODELS):
        out["modules_categorized"] = 1.0

    return out
