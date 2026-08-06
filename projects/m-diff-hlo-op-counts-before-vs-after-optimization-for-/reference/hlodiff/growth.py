def measure_growth(sizes):
    """Measure HLO dump size and line count growth across model sizes."""
    results = []
    ops = ["add", "multiply", "dot", "reshape", "transpose"]
    for s in sizes:
        text = "HloModule model_" + str(s) + "\nENTRY main {\n"
        for i in range(s):
            op = ops[i % len(ops)]
            text += f"  {op}_{i} = f32[32,32] {op}(p0, p1)\n"
        text += "  ROOT root = f32[32,32] copy(" + f"{ops[-1]}_{s-1}" + ")\n}\n"
        results.append({"size": s, "bytes": len(text.encode("utf-8")), "line_count": len(text.splitlines())})
    return results
