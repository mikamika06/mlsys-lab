import ref

def check(workdir):
    from runner.raw import prove_raw_mode
    ok = 0
    for transcript, expected in ref.TRANSCRIPTS:
        try:
            res = prove_raw_mode(transcript)
            if res == expected:
                ok += 1
        except Exception:
            pass
    passed = 1.0 if ok == len(ref.TRANSCRINTS if hasattr(ref, "TRANSCRINTS") else ref.TRANSCRIPTS) else 0.0
    return {"raw_proven_match": 1.0 if ok == len(ref.TRANSCRIPTS) else 0.0}
