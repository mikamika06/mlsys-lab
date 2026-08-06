def measure_trace_overhead(traces_no_stack, traces_with_stack):
    size_no = sum(len(t) for t in traces_no_stack)
    size_with = sum(len(t) for t in traces_with_stack)
    if size_no == 0:
        return 0.0
    return float(size_with) / float(size_no)
