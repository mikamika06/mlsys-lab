class MockSession:
    def __init__(self, kind, time_state):
        self.kind = kind
        self.time_state = time_state
        self.time_state["now"] += (0.5 if kind == "GPU" else 0.1)
        self.runs = 0

    def run(self, inputs):
        if self.runs == 0:
            self.time_state["now"] += (1.5 if self.kind == "GPU" else 0.2)
        else:
            self.time_state["now"] += (0.01 if self.kind == "GPU" else 0.05)
        self.runs += 1
        return {"out": 1}

def check_shadow_wheel(installed_packages):
    return "onnxruntime" in installed_packages and "onnxruntime-gpu" in installed_packages

def measure_breakdown(session_factory, inputs, time_fn, num_steady=5):
    t0 = time_fn()
    sess = session_factory()
    t1 = time_fn()
    sess.run(inputs)
    t2 = time_fn()
    for _ in range(num_steady):
        sess.run(inputs)
    t3 = time_fn()
    return {
        "creation": t1 - t0,
        "first_run": t2 - t1,
        "steady_step": (t3 - t2) / num_steady
    }

def ep_delta(cpu_factory, alt_factory, inputs, time_fn, num_steady=5):
    cpu_perf = measure_breakdown(cpu_factory, inputs, time_fn, num_steady)
    alt_perf = measure_breakdown(alt_factory, inputs, time_fn, num_steady)
    return {
        "steady_speedup": cpu_perf["steady_step"] / alt_perf["steady_step"],
        "cold_penalty": (alt_perf["creation"] + alt_perf["first_run"]) - (cpu_perf["creation"] + cpu_perf["first_run"])
    }
