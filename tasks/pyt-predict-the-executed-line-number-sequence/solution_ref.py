import sys


def predict_line_sequence(fn):
    target = fn.__code__
    result = []

    def tracer(frame, event, arg):
        if frame.f_code is target and event == "line":
            result.append(frame.f_lineno)
        return tracer

    sys.settrace(tracer)
    try:
        fn()
    finally:
        sys.settrace(None)

    return result
