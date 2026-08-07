def check(workdir):
    from pte import plan
    import ref

    m = {"keys_ok": 0.0, "no_overlap": 0.0, "saved_mem": 0.0}
    raw = ref.get_large_data()
    try:
        tensors = plan.parse_artifact(raw)
        c, a = plan.split_program_data(tensors)
        allocs = plan.replan_buffers(a)
        if len(allocs) == len(a):
            m["keys_ok"] = 1.0

        overlap_found = False
        for t1 in a:
            for t2 in a:
                if t1["id"] >= t2["id"]:
                    continue
                l_overlap = t1["start"] < t2["end"] and t2["start"] < t1["end"]
                if l_overlap:
                    o1, o2 = allocs[t1["id"]], allocs[t2["id"]]
                    m_overlap = not (o1 + t1["size"] <= o2 or o2 + t2["size"] <= o1)
                    if m_overlap:
                        overlap_found = True

        if not overlap_found:
            m["no_overlap"] = 1.0

        peak = max(allocs[t["id"]] + t["size"] for t in a)
        if peak <= 200:
            m["saved_mem"] = 1.0
    except Exception:
        pass
    return m
