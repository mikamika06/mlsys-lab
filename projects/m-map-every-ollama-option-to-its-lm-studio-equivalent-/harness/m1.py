import ref


def check(workdir):
    from runner_map.mapping import map_option
    out = {"mapping_match": 0.0}
    ok = 0
    total = len(ref.OLLAMA_OPTIONS)
    for opt, expected in ref.OLLAMA_OPTIONS:
        got = map_option(opt)
        if got == expected:
            ok += 1
    if ok == total:
        out["mapping_match"] = 1.0
    else:
        out["_note"] = f"Matched {ok}/{total} options correctly."
    return out
