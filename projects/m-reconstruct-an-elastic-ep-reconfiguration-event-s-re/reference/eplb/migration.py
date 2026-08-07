import numpy as np


def plan_migration(old_layout, target_layout):
    actions = []
    num_ranks = len(old_layout)
    old_flat = {}
    for r, exps in old_layout.items():
        for e in exps:
            old_flat.setdefault(e, []).append(r)

    target_flat = {}
    for r, exps in target_layout.items():
        for e in exps:
            target_flat.setdefault(e, []).append(r)

    for e in sorted(set(list(old_flat.keys()) + list(target_flat.keys()))):
        o_ranks = list(old_flat.get(e, []))
        t_ranks = list(target_flat.get(e, []))

        for r in o_ranks[:]:
            if r in t_ranks:
                o_ranks.remove(r)
                t_ranks.remove(r)

        while o_ranks and t_ranks:
            sr = o_ranks.pop(0)
            tr = t_ranks.pop(0)
            actions.append({"expert": e, "from_rank": sr, "to_rank": tr})

        for sr in o_ranks:
            actions.append({"expert": e, "from_rank": sr, "to_rank": None})

        for tr in t_ranks:
            actions.append({"expert": e, "from_rank": None, "to_rank": tr})

    return actions
