import ref

def check(workdir):
    from profdebug.disambiguate import disambiguate
    ok = 0
    for transcript, expected in ref.TRANSCRIPTS:
        if disambiguate(transcript) == expected:
            ok += 1
    return {"disambiguate_match": float(ok)}
