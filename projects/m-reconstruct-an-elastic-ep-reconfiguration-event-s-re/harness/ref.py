def apply_event(layout, event):
    new_layout = {k: list(v) for k, v in layout.items()}
    kind = event.get("type")
    exp = event.get("expert")

    if kind == "add":
        rank = event["rank"]
        if rank in new_layout and exp not in new_layout[rank]:
            new_layout[rank].append(exp)
    elif kind == "remove":
        rank = event["rank"]
        if rank in new_layout and exp in new_layout[rank]:
            new_layout[rank].remove(exp)
    elif kind == "move":
        src = event["source"]
        dst = event["dest"]
        if src in new_layout and exp in new_layout[src]:
            new_layout[src].remove(exp)
        if dst in new_layout and exp not in new_layout[dst]:
            new_layout[dst].append(exp)

    for k in new_layout:
        new_layout[k].sort()

    return new_layout


def reconstruct_layout(initial_layout, events):
    current = initial_layout
    for ev in events:
        current = apply_event(current, ev)
    return current


SINGLE_EVENTS = [
    ({"type": "add", "expert": 5, "rank": 1}, {0: [1], 1: [2]}),
    ({"type": "remove", "expert": 2, "rank": 1}, {0: [1], 1: [2]}),
    ({"type": "move", "expert": 1, "source": 0, "dest": 1}, {0: [1], 1: [2]})
]

BATCH_EVENTS = [
    (
        {0: [0, 1], 1: [2, 3], 2: [4, 5]},
        [
            {"type": "move", "expert": 1, "source": 0, "dest": 2},
            {"type": "remove", "expert": 3, "rank": 1},
            {"type": "add", "expert": 0, "rank": 1}
        ]
    ),
    (
        {0: [0, 2], 1: [1]},
        [
            {"type": "add", "expert": 2, "rank": 1},
            {"type": "move", "expert": 2, "source": 0, "dest": 1},
            {"type": "remove", "expert": 2, "rank": 0}
        ]
    )
]
