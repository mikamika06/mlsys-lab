import sys


def _oracle():
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
    wanted = ("number", "text", "cell_value")
    locals_out = {name: frame.f_locals[name] for name in wanted}
    cell_out = {
        name: frame.f_locals[name]
        for name in frame.f_code.co_cellvars
        if name in frame.f_locals
    }
    return {"locals": locals_out, "cellvars": cell_out}


def grade(sol, fx) -> dict:
    try:
        got = sol.read_frame_locals_and_cellvars()
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0 if got == _oracle() else 0.0}
