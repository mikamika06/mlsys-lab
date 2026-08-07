class Tracer:
    """Trace function executions for compilation profiling."""
    def trace(self, func, inputs):
        results = []
        for x in inputs:
            results.append(func(x))
        return results
