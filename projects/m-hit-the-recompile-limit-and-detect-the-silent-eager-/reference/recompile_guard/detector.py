class RecompileDetector:
    def __init__(self, limit=8):
        self.limit = limit
        self._compiles = {}
        self._executions = {}

    def register(self, fn_name):
        if fn_name not in self._compiles:
            self._compiles[fn_name] = 0
            self._executions[fn_name] = 0

    def record_compile(self, fn_name):
        self.register(fn_name)
        self._compiles[fn_name] += 1

    def record_execution(self, fn_name):
        self.register(fn_name)
        self._executions[fn_name] += 1

    def is_limit_exceeded(self, fn_name):
        return self._compiles.get(fn_name, 0) >= self.limit

    def is_silent_fallback(self, fn_name):
        c = self._compiles.get(fn_name, 0)
        e = self._executions.get(fn_name, 0)
        return c >= self.limit and e > c

    def get_stats(self, fn_name):
        c = self._compiles.get(fn_name, 0)
        e = self._executions.get(fn_name, 0)
        exceeded = c >= self.limit
        fallback = exceeded and e > c
        return {
            "compiles": c,
            "executions": e,
            "limit": self.limit,
            "limit_exceeded": exceeded,
            "silent_fallback": fallback,
        }
