import ref


def check(workdir):
    from cudagraphs.safety import analyze_graph_safety

    ops = ref.generate_safety_operations()
    want = ref.analyze_safety_ref(ops)

    try:
        got = analyze_graph_safety(ops)
    except Exception as e:
        return {"classifications_correct": 0.0, "_note": str(e)}

    correct = 1.0 if got == want else 0.0
    out = {"classifications_correct": correct}
    if not correct:
        out["_note"] = f"Expected {want}, got {got}"
    return out
