import sys
import os
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from prompt_lookup.draft import draft_ngram
    except ImportError:
        return {"_note": "could not import draft_ngram from prompt_lookup.draft"}

    out = {"matches": 0.0, "total": float(len(ref.M1_CASES))}
    ok = 0
    for tokens, max_n, max_draft_len, want in ref.M1_CASES:
        try:
            got = draft_ngram(tokens, max_n, max_draft_len)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"got {got}, want {want} for {tokens[-5:]}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"crashed: {type(e).__name__}: {str(e)}"

    out["matches"] = float(ok)
    sys.path.pop(0)
    return out
