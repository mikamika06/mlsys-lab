import numpy as np

_MOD = 3
_TRANSITIONS = {}
for _r in range(_MOD):
    _TRANSITIONS[(_r, '0')] = (2 * _r) % _MOD
    _TRANSITIONS[(_r, '1')] = (2 * _r + 1) % _MOD
_START = 0
_ACCEPT = {0}


def _oracle_run(s):
    """Independent reference walk: `s` is accepted iff it is the binary
    encoding of a non-negative integer divisible by 3 (empty string -> 0,
    accepted)."""
    state = _START
    for ch in s:
        key = (state, ch)
        if key not in _TRANSITIONS:
            return False
        state = _TRANSITIONS[key]
    return state in _ACCEPT


def grade(sol, fx) -> dict:
    strings = [str(x) for x in fx["strings"].tolist()]
    ref = np.array([_oracle_run(s) for s in strings], dtype=np.int64)

    got_list = []
    for s in strings:
        try:
            got = bool(sol.run_fsm(dict(_TRANSITIONS), _START, set(_ACCEPT), s))
        except Exception:
            return {"exact_match": 0.0}
        got_list.append(got)

    got_arr = np.array(got_list, dtype=np.int64)
    ok = 1.0 if np.array_equal(got_arr, ref) else 0.0
    return {"exact_match": ok}
