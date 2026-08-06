import ref

def check(workdir):
    from jsonfsm.fsm import JSONFSM
    out = {"fsm_match": 0.0}
    fsm = JSONFSM(ref.SCHEMAS[0])
    toks = fsm.allowed_tokens(ref.VOCAB)
    if isinstance(toks, list) and len(toks) > 0:
        out["fsm_match"] = 1.0
    return out
