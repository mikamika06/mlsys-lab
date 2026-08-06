class KernelAttributionHarness:
    def __init__(self):
        self.events = []

    def register_trace(self, trace_events):
        self.events = trace_events

    def attribute_kernels(self):
        attributed = []
        stack = []
        for event in sorted(self.events, key=lambda x: x.get("ts", 0)):
            ev_type = event.get("ph")
            if ev_type == "B":
                stack.append(event)
            elif ev_type == "E":
                if stack:
                    stack.pop()
            elif ev_type == "X":
                ts = event.get("ts", 0)
                dur = event.get("dur", 0)
                cat = event.get("cat", "")
                if cat == "kernel":
                    parent_scope = stack[-1]["name"] if stack else "root"
                    attributed.append({
                        "name": event.get("name"),
                        "scope": parent_scope,
                        "dur": dur,
                        "ts": ts
                    })
        return attributed
