import ref

def check(workdir):
    out = {"attributed_events": 0.0}
    try:
        from kernel_attr.attribution import KernelAttributionHarness
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    harness = KernelAttributionHarness()
    harness.register_trace(ref.TRACE_EVENTS)
    got = harness.attribute_kernels()
    expected = ref.ref_attribute_kernels(ref.TRACE_EVENTS)

    if len(got) == len(expected):
        match = True
        for g, e in zip(got, expected):
            if g.get("name") != e.get("name") or g.get("scope") != e.get("scope") or g.get("dur") != e.get("dur"):
                match = False
                break
        if match:
            out["attributed_events"] = 1.0
        else:
            out["_note"] = f"Mismatch in attributed kernels: got {got}, expected {expected}"
    else:
        out["_note"] = f"Length mismatch: got {len(got)} events, expected {len(expected)}"

    return out
