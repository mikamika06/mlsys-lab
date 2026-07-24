import sys
import inspect

def _trace_count(cls, K=5):
    obj = cls([1, 2, 3])
    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        if frame.f_code.co_name == 'sum':
            start = frame.f_code.co_firstlineno
            source_lines, _ = inspect.getsourcelines(frame.f_code)
            end = start + len(source_lines) - 1
            if start <= frame.f_lineno <= end:
                count += 1
        return tracer

    sys.settrace(tracer)
    for _ in range(K):
        _ = obj.sum
    sys.settrace(None)
    return count

def grade(sol, fx) -> dict:
    class OracleRecomputeCounter:
        def __init__(self, data):
            self.data = list(data)

        @property
        def sum(self):
            return sum(self.data)

    ref_count = _trace_count(OracleRecomputeCounter)
    try:
        cls = getattr(sol, 'RecomputeCounter')
    except AttributeError:
        return {"exact_match": 0.0}
    cand_count = _trace_count(cls)
    ok = 1.0 if cand_count == ref_count else 0.0
    return {"exact_match": ok}
