import sys
import types


def _walk(tb):
    out = []
    while tb is not None:
        out.append((tb.tb_frame.f_code.co_name, tb.tb_lineno))
        tb = tb.tb_next
    return out


def _oracle(frames):
    tb = None
    for frame, lineno in reversed(frames):
        tb = types.TracebackType(tb, frame, frame.f_lasti, lineno)
    return _walk(tb)


def _make_frames():
    def child():
        frame = sys._getframe()
        return frame

    def parent():
        parent_frame = sys._getframe()
        child_frame = child()
        return [
            (parent_frame, parent_frame.f_lineno),
            (child_frame, child_frame.f_lineno),
        ]

    return parent()


def grade(sol, fx) -> dict:
    cases = [
        _make_frames(),
        _make_frames(),
        _make_frames(),
    ]

    ok = 1.0
    for frames in cases:
        try:
            ref = _oracle(frames)
            got = sol.reconstruct_traceback(frames)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
