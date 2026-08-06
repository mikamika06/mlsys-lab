import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)

    try:
        from ovdev.selection import inspect_auto_allocations
        import ref as harness_ref
    except Exception as e:
        return {"selections_matched": 0.0, "_note": f"Import error: {e}"}

    targets = harness_ref.generate_targets()
    
    # Compute reference
    ref_results = []
    for t in targets:
        props = t["properties"]
        hint = t["hint"]
        exec_devs = props.get("EXECUTION_DEVICES", [])
        if exec_devs:
            actual = exec_devs[0]
        else:
            avail = props.get("AVAILABLE_DEVICES", [])
            if hint == "THROUGHPUT" and "GPU" in avail:
                actual = "GPU"
            elif hint == "LATENCY" and "NPU" in avail:
                actual = "NPU"
            else:
                actual = avail[0] if avail else "CPU"
        
        ref_results.append({
            "target_id": t["id"],
            "requested_device": t.get("device", "AUTO"),
            "actual_device": actual,
            "is_fallback": actual != t.get("preferred_device", actual)
        })

    try:
        got_results = inspect_auto_allocations(targets)
    except Exception as e:
        return {"selections_matched": 0.0, "_note": f"Execution error: {e}"}

    if got_results == ref_results:
        return {"selections_matched": 1.0}
    
    return {
        "selections_matched": 0.0,
        "_note": f"Mismatch. Expected {ref_results[:2]}, got {got_results[:2]}"
    }
