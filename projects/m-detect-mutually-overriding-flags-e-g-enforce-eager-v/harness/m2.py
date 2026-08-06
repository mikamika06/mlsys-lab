import ref


def check(workdir):
    from vllmargs.translate import cli_to_engine, engine_to_cli

    out = {"translation_matched": 0.0}
    ok = 0
    total = len(ref.TRANSLATION_CASES) * 2
    for cli_in, eng_in in ref.TRANSLATION_CASES:
        got_eng = cli_to_engine(cli_in)
        got_cli = engine_to_cli(eng_in)
        if got_eng == eng_in:
            ok += 1
        if got_cli == cli_in:
            ok += 1
    out["translation_matched"] = 1.0 if ok == total else 0.0
    if ok != total and "_note" not in out:
        out["_note"] = f"matched {ok} of {total} translation directions"
    return out
