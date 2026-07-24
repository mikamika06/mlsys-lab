import sys


def set_allocation_trace(values):
    s = set()
    trace = []
    for value in values:
        s.add(value)
        trace.append(sys.getsizeof(s))
    return trace
