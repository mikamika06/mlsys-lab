import ref

def check(workdir):
    from compilebench.breakfinder import identify_graph_break
    logs = ["Graph break triggered at line 15 due to dynamic control flow"]
    res = identify_graph_break(logs)
    matched = 1.0 if ("15" in res or "dynamic" in res) else 0.0
    return {"breaks_identified": float(matched)}
