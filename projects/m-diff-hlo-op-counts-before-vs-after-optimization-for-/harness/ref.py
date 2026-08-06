import json

SAMPLE_BEFORE_1 = """
HloModule test1
ENTRY main.1 {
  p0 = f32[128,128] parameter(0)
  p1 = f32[128,128] parameter(1)
  add = f32[128,128] add(p0, p1)
  relu = f32[128,128] maximum(add, 0)
  ROOT mul = f32[128,128] multiply(relu, p0)
}
"""

SAMPLE_AFTER_1 = """
HloModule test1
ENTRY main.1 {
  p0 = f32[128,128] parameter(0)
  p1 = f32[128,128] parameter(1)
  fusion = f32[128,128] fusion(p0, p1), kind=kLoop, calls=fusion_computation
}
"""

SAMPLE_FUSION_HLO = """
HloModule fusion_test
fusion_computation {
  arg0 = f32[64] parameter(0)
  arg1 = f32[64] parameter(1)
  add.1 = f32[64] add(arg0, arg1)
  ROOT mul.1 = f32[64] multiply(add.1, arg0)
}
ENTRY main {
  p0 = f32[64] parameter(0)
  p1 = f32[64] parameter(1)
  ROOT f_out = f32[64] fusion(p0, p1), kind=kInput, calls=fusion_computation
}
"""

def generate_mock_dumps(model_size):
    ops = ["add", "multiply", "dot", "reshape", "transpose"]
    text = "HloModule model_" + str(model_size) + "\nENTRY main {\n"
    for i in range(model_size):
        op = ops[i % len(ops)]
        text += f"  {op}_{i} = f32[32,32] {op}(p0, p1)\n"
    text += "  ROOT root = f32[32,32] copy(" + f"{ops[-1]}_{model_size-1}" + ")\n}\n"
    return text

TEST_CASES = [
    (SAMPLE_BEFORE_1, SAMPLE_AFTER_1),
    (generate_mock_dumps(10), generate_mock_dumps(5)),
    (generate_mock_dumps(20), generate_mock_dumps(12)),
]

def diff_op_counts(before_text, after_text):
    def parse_ops(text):
        counts = {}
        for line in text.splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("HloModule") and not line.startswith("ENTRY") and "{" not in line and "}" not in line:
                parts = line.split("=")
                if len(parts) == 2:
                    rhs = parts[1].strip()
                    op_name = rhs.split("(")[0].strip().split()[0]
                    counts[op_name] = counts.get(op_name, 0) + 1
        return counts
    b_counts = parse_ops(before_text)
    a_counts = parse_ops(after_text)
    all_keys = set(b_counts.keys()).union(set(a_counts.keys()))
    delta = {}
    for k in all_keys:
        delta[k] = a_counts.get(k, 0) - b_counts.get(k, 0)
    return {"before": b_counts, "after": a_counts, "delta": delta}

def count_fusion_kernels(hlo_text):
    kernels = []
    for line in hlo_text.splitlines():
        line = line.strip()
        if "fusion(" in line or "fusion =" in line:
            if "calls=" in line:
                call_part = line.split("calls=")[1].strip().split(",")[0].strip()
                kernels.append(call_part)
            else:
                kernels.append("unnamed_fusion")
    return {"fusion_count": len(kernels), "kernel_names": sorted(list(set(kernels)))}

def measure_growth(sizes):
    results = []
    for s in sizes:
        dump = generate_mock_dumps(s)
        results.append({"size": s, "bytes": len(dump.encode("utf-8")), "line_count": len(dump.splitlines())})
    return results
