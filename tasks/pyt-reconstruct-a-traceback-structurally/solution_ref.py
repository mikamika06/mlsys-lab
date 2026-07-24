import types


def reconstruct_traceback(frames):
    tb = None
    for frame, lineno in reversed(frames):
        tb = types.TracebackType(tb, frame, frame.f_lasti, lineno)

    walk = []
    while tb is not None:
        walk.append((tb.tb_frame.f_code.co_name, tb.tb_lineno))
        tb = tb.tb_next
    return walk
