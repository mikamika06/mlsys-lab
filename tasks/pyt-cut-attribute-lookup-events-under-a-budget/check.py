import dis
import sys


class Box:
    pass


def _make_model(value):
    model = Box()
    model.state = Box()
    model.state.block = Box()
    model.state.block.weight = Box()
    model.state.block.weight.value = value
    return model


def _oracle(model, steps):
    total = 0
    for _ in range(steps):
        total += model.state.block.weight.value
    return total


def _count_load_attr(fn, model, steps):
    offsets = {
        ins.offset
        for ins in dis.get_instructions(fn)
        if ins.opname == "LOAD_ATTR"
    }
    count = 0

    def trace(frame, event, arg):
        nonlocal count
        if event == "call" and frame.f_code is fn.__code__:
            frame.f_trace_opcodes = True
            return trace
        if event == "opcode" and frame.f_code is fn.__code__:
            if frame.f_lasti in offsets:
                count += 1
        return trace

    sys.settrace(trace)
    try:
        fn(model, steps)
    finally:
        sys.settrace(None)
    return count


def grade(sol, fx) -> dict:
    cases = [
        (2, 1),
        (3, 5),
        (7, 50),
        (13, 200),
    ]

    exact = 1.0
    attr_ok = 1.0

    for value, steps in cases:
        model = _make_model(value)
        try:
            got = sol.accumulate_metric(model, steps)
            ref = _oracle(model, steps)
            events = _count_load_attr(sol.accumulate_metric, model, steps)
        except Exception:
            exact = 0.0
            attr_ok = 0.0
            break

        if got != ref:
            exact = 0.0

        if events > 5:
            attr_ok = 0.0

    return {
        "exact_match": exact,
        "load_attr_events": attr_ok,
    }
