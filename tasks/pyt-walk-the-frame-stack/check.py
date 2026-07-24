import sys


def _oracle_chain():
    frame = sys._getframe().f_back
    out = []
    while frame is not None:
        out.append(frame.f_code.co_qualname)
        frame = frame.f_back
    return out


def _deep_call(fn):
    return fn()


def grade(sol, fx) -> dict:
    try:
        expected = _deep_call(_oracle_chain)
        got = _deep_call(sol.frame_qualname_chain)
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == expected else 0.0}
