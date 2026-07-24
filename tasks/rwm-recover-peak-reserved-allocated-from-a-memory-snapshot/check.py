def _oracle(snapshot):
    reserved = sum(segment["size"] for segment in snapshot["segments"])
    active = {}
    current_allocated = 0
    peak_reserved = 0
    peak_allocated = 0
    largest = 0

    events = []
    for segment in snapshot["segments"]:
        for block in segment["blocks"]:
            for event in block["events"]:
                events.append((event, block))

    for event, block in events:
        if event == "alloc":
            active[block["id"]] = True
            current_allocated += block["size"]
            largest = max(largest, block["size"])
        else:
            del active[block["id"]]
            current_allocated -= block["size"]

        peak_reserved = max(peak_reserved, reserved)
        peak_allocated = max(peak_allocated, current_allocated)

    return (peak_reserved, peak_allocated, largest)


def grade(sol, fx) -> dict:
    cases = [
        {
            "segments": [
                {
                    "id": "s0",
                    "size": 1024,
                    "blocks": [
                        {"id": "a", "size": 128, "events": ["alloc", "free"]},
                        {"id": "b", "size": 256, "events": ["alloc"]},
                    ],
                }
            ]
        },
        {
            "segments": [
                {
                    "id": "s0",
                    "size": 400,
                    "blocks": [
                        {"id": "x", "size": 50, "events": ["alloc", "free", "alloc", "free"]},
                    ],
                },
                {
                    "id": "s1",
                    "size": 700,
                    "blocks": [
                        {"id": "y", "size": 300, "events": ["alloc"]},
                        {"id": "z", "size": 100, "events": ["alloc", "free"]},
                    ],
                },
            ]
        },
        {
            "segments": [
                {
                    "id": "s0",
                    "size": 2048,
                    "blocks": [
                        {"id": "a", "size": 512, "events": ["alloc", "free"]},
                        {"id": "b", "size": 1024, "events": ["alloc", "free"]},
                        {"id": "c", "size": 256, "events": ["alloc"]},
                    ],
                }
            ]
        },
    ]

    ok = 1.0
    for snapshot in cases:
        try:
            got = sol.recover_memory_peaks(snapshot)
        except Exception:
            ok = 0.0
            break
        if tuple(got) != _oracle(snapshot):
            ok = 0.0
            break
    return {"exact_match": ok}
