import dis
import sys


def count_attribute_lookups(cls):
    target = dis.opmap["LOAD_ATTR"]
    count = 0
    old_trace = sys.gettrace()

    def tracer(frame, event, arg):
        nonlocal count
        if event == "call":
            frame.f_trace_opcodes = True
            return tracer
        if event == "opcode":
            if frame.f_lasti >= 0 and frame.f_code.co_code[frame.f_lasti] == target:
                count += 1
        return tracer

    sys.settrace(tracer)
    try:
        cls()
    finally:
        sys.settrace(old_trace)
    return count
