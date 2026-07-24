import dis
import sys


class _Probe:
    created = 0

    def __new__(cls):
        obj = super().__new__(cls)
        obj.slot = cls.created
        return obj

    def __init__(self):
        self.slot = self.slot + 1
        self.extra = self.__class__.__name__


def _oracle_count(cls):
    target = dis.opmap["LOAD_ATTR"]
    count = 0
    old_trace = sys.gettrace()

    def tracer(frame, event, arg):
        if event == "call":
            frame.f_trace_opcodes = True
            return tracer
        if event == "opcode":
            nonlocal count
            code = frame.f_code
            if frame.f_lasti >= 0 and code.co_code[frame.f_lasti] == target:
                count += 1
        return tracer

    sys.settrace(tracer)
    try:
        cls()
    finally:
        sys.settrace(old_trace)
    return count


def grade(sol, fx) -> dict:
    expected = _oracle_count(_Probe)
    try:
        got = sol.count_attribute_lookups(_Probe)
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": float(got == expected)}
