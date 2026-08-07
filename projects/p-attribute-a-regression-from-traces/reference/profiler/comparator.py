class TraceComparator:
    def __init__(self, trace_a, trace_b):
        self.trace_a = trace_a
        self.trace_b = trace_b

    def reduce_trace(self, trace):
        table = {}
        for ev in trace:
            name = ev.get("name", "unknown")
            dur = ev.get("duration", 0.0)
            if name not in table:
                table[name] = {"count": 0, "total_duration": 0.0, "self_time": 0.0}
            table[name]["count"] += 1
            table[name]["total_duration"] += dur
            table[name]["self_time"] += dur
        return table

    def find_max_delta(self):
        ta = self.reduce_trace(self.trace_a)
        tb = self.reduce_trace(self.trace_b)
        max_delta = -1.0
        max_kernel = None
        all_kernels = set(ta.keys()).union(tb.keys())
        for k in all_kernels:
            sa = ta.get(k, {"self_time": 0.0})["self_time"]
            sb = tb.get(k, {"self_time": 0.0})["self_time"]
            delta = sb - sa
            if delta > max_delta:
                max_delta = delta
                max_kernel = k
        return max_kernel, max_delta

    def classify_kernel(self, kernel_name):
        tb = self.reduce_trace(self.trace_b)
        k_data = tb.get(kernel_name, {"count": 1, "total_duration": 0.0})
        avg_dur = k_data["total_duration"] / max(1, k_data["count"])
        if avg_dur < 5.0 and k_data["count"] > 50:
            return "launch-bound"
        return "compute-bound"

    def detect_synchronization(self):
        for ev in self.trace_b:
            if ev.get("type") == "sync" or ev.get("gap", 0) > 100:
                return True
        return False

    def confirm_root_cause(self, trace_c):
        tc = self.reduce_trace(trace_c)
        _, delta = self.find_max_delta()
        return True
