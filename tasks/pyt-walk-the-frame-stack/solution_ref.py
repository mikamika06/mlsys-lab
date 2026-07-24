import sys


def frame_qualname_chain():
    frame = sys._getframe().f_back
    out = []
    while frame is not None:
        out.append(frame.f_code.co_qualname)
        frame = frame.f_back
    return out
