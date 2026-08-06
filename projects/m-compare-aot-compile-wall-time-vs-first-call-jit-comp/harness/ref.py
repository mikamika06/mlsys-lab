import time
import re

class MockExecutable:
    def __init__(self, name, delay=0.002):
        self.name = name
        self.delay = delay

    def __call__(self, *args):
        time.sleep(self.delay)
        return 1.0

class MockLowered:
    def __init__(self, name, ops_list, compile_delay=0.008):
        self.name = name
        self.ops_list = ops_list
        self.compile_delay = compile_delay

    def compile(self):
        time.sleep(self.compile_delay)
        return MockExecutable(self.name)

    def as_text(self, dialect="stablehlo"):
        lines = [f"module @{self.name} {{", "  func.func @main() {"]
        for op in self.ops_list:
            lines.append(f'    %res = "stablehlo.{op}"(%arg) : () -> ()')
        lines.extend(["    return %res", "  }", "}"])
        return "\n".join(lines)

class MockJitFunction:
    def __init__(self, name="sample_fn", base_ops=None, flag_transforms=None, compile_delay=0.008, exec_delay=0.002, trace_delay=0.004):
        self.name = name
        self.base_ops = list(base_ops if base_ops is not None else ["dot", "add", "add", "multiply"])
        self.flag_transforms = flag_transforms or {}
        self.compile_delay = compile_delay
        self.exec_delay = exec_delay
        self.trace_delay = trace_delay
        self.is_compiled = False

    def fresh_instance(self):
        return MockJitFunction(
            self.name,
            self.base_ops,
            self.flag_transforms,
            self.compile_delay,
            self.exec_delay,
            self.trace_delay,
        )

    def lower(self, *args, flags=None):
        ops = list(self.base_ops)
        flags = flags or {}
        for flag_key, flag_val in flags.items():
            if flag_key in self.flag_transforms:
                ops = self.flag_transforms[flag_key](ops, flag_val)
        return MockLowered(self.name, ops, compile_delay=self.compile_delay)

    def __call__(self, *args):
        if not self.is_compiled:
            time.sleep(self.trace_delay + self.compile_delay)
            self.is_compiled = True
        time.sleep(self.exec_delay)
        return 1.0

def compare_compilation_timings(jit_fn, *args):
    target_aot = jit_fn.fresh_instance() if hasattr(jit_fn, "fresh_instance") else jit_fn
    lowered = target_aot.lower(*args)

    t0 = time.perf_counter()
    _ = lowered.compile()
    t1 = time.perf_counter()
    aot_compile_time = t1 - t0

    target_jit = jit_fn.fresh_instance() if hasattr(jit_fn, "fresh_instance") else jit_fn
    t2 = time.perf_counter()
    _ = target_jit(*args)
    t3 = time.perf_counter()
    jit_first_call_time = t3 - t2

    t4 = time.perf_counter()
    _ = target_jit(*args)
    t5 = time.perf_counter()
    jit_cached_time = t5 - t4

    denom = max(jit_first_call_time - jit_cached_time, 1e-9)
    overhead_ratio = aot_compile_time / denom

    return {
        "aot_compile_time": float(aot_compile_time),
        "jit_first_call_time": float(jit_first_call_time),
        "jit_cached_time": float(jit_cached_time),
        "overhead_ratio": float(overhead_ratio),
    }

def parse_stablehlo_op_counts(stablehlo_code):
    matches = re.findall(r"stablehlo\.([a-zA-Z0-9_]+)", stablehlo_code)
    counts = {}
    for op in matches:
        counts[op] = counts.get(op, 0) + 1
    return counts

def get_stablehlo_ops(jit_fn, args, flags=None):
    flags = flags or {}
    try:
        lowered = jit_fn.lower(*args, flags=flags)
    except TypeError:
        lowered = jit_fn.lower(*args)
    text = lowered.as_text(dialect="stablehlo")
    return parse_stablehlo_op_counts(text)

def diff_stablehlo_ops(jit_fn, args, flags_a=None, flags_b=None):
    ops_a = get_stablehlo_ops(jit_fn, args, flags_a)
    ops_b = get_stablehlo_ops(jit_fn, args, flags_b)
    all_ops = sorted(set(ops_a.keys()) | set(ops_b.keys()))
    return {op: ops_b.get(op, 0) - ops_a.get(op, 0) for op in all_ops}
