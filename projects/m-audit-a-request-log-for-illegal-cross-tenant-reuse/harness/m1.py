import ref


def check(workdir):
    from logaudit.tracker import CacheTracker

    out = {"traces_parsed": 0.0, "block_mappings_correct": 0.0}
    logs = ref.generate_traces(seed=101)

    try:
        tracker = CacheTracker(block_size=16)
        expected_owners = {}
        for event in logs:
            tracker.process_event(event)
            if event["type"] == "allocate":
                expected_owners[event["block_id"]] = event["tenant_id"]

        out["traces_parsed"] = 1.0
        got_owners = tracker.get_block_owners()
        if got_owners == expected_owners:
            out["block_mappings_correct"] = 1.0
        else:
            out["_note"] = f"Expected {len(expected_owners)} owners, got {len(got_owners)}"
    except Exception as e:
        out["_note"] = f"Failed to parse trace: {type(e).__name__}: {e}"

    return out
