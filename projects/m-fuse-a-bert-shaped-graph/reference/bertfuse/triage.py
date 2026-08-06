def triage_attention(graph):
    unfused = [n["name"] for n in graph["nodes"] if n["op"] in ("MatMul", "Add") and "Q" in n["name"]]
    return {"unfused_count": len(unfused), "status": "triaged" if len(unfused) == 0 else "unfused"}
