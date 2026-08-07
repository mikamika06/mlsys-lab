import ref

def check(workdir):
    from runner.chat import continue_conversation
    ok = 0
    for context, messages, expected in ref.CONTEXTS:
        try:
            res = continue_conversation(context, messages)
            if res == expected:
                ok += 1
        except Exception:
            pass
    return {"context_state_match": 1.0 if ok == len(ref.CONTEXTS) else 0.0}
