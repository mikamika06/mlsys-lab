import sys


def read_frame_locals_and_cellvars():
    captured = {}

    def target():
        number = 42
        text = "frame"
        cell_value = 7

        def inner():
            return cell_value

        probe = True
        return inner

    def tracer(frame, event, arg):
        if event == "line" and frame.f_code.co_name == "target" and "probe" in frame.f_locals:
            captured["frame"] = frame
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        target()
    finally:
        sys.settrace(old)

    frame = captured["frame"]
    names = ("number", "text", "cell_value")
    return {
        "locals": {name: frame.f_locals[name] for name in names},
        "cellvars": {
            name: frame.f_locals[name]
            for name in frame.f_code.co_cellvars
            if name in frame.f_locals
        },
    }
