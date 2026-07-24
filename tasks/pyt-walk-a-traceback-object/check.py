import os


def _oracle(exc):
    out = []
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        out.append((os.path.basename(frame.f_code.co_filename), tb.tb_lineno, frame.f_code.co_name))
        tb = tb.tb_next
    return out


def _make_case_one():
    def level_two():
        raise RuntimeError("case one")

    def level_one():
        level_two()

    level_one()


def _make_case_two():
    def branch():
        raise KeyError("case two")

    branch()


def grade(sol, fx) -> dict:
    cases = [_make_case_one, _make_case_two]
    ok = 1.0

    for case in cases:
        try:
            case()
        except Exception as exc:
            expected = _oracle(exc)
            try:
                got = sol.walk_traceback(exc)
            except Exception:
                ok = 0.0
                break
            if got != expected:
                ok = 0.0
                break
        else:
            ok = 0.0
            break

    return {"exact_match": ok}
