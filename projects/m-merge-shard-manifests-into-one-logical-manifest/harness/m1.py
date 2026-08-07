import ref


def check(workdir):
    from shards.manifest import validate_filename

    case = ref.get_test_cases()
    ok = 0
    total = len(case["filenames_good"]) + len(case["filenames_bad"])

    for f in case["filenames_good"]:
        if validate_filename(f) is True:
            ok += 1
    for f in case["filenames_bad"]:
        if validate_filename(f) is False:
            ok += 1

    out = {"filenames_matched": float(ok)}
    if ok < total:
        out["_note"] = f"Matched {ok} out of {total} filename test cases correctly."
    return out
