from collections import deque


def diagnose_schema(masker) -> dict:
    """Diagnoses whether a schema state machine is satisfiable."""
    visited = set()
    queue = deque([0])
    reachable_states = set()

    while queue:
        curr = queue.popleft()
        if curr in visited:
            continue
        visited.add(curr)
        reachable_states.add(curr)
        transitions = masker.transitions.get(curr, {})
        for tid, nxt in transitions.items():
            if nxt not in visited:
                queue.append(nxt)

    can_reach_terminal = set()
    for s in reachable_states:
        st_visited = set()
        st_queue = deque([s])
        found_term = False
        while st_queue:
            node = st_queue.popleft()
            if node in masker.terminal_states:
                found_term = True
                break
            if node in st_visited:
                continue
            st_visited.add(node)
            for tid, nxt in masker.transitions.get(node, {}).items():
                if nxt not in st_visited:
                    st_queue.append(nxt)
        if found_term:
            can_reach_terminal.add(s)

    deadlock_states = sorted(list(reachable_states - can_reach_terminal))
    is_satisfiable = 0 in can_reach_terminal

    return {
        "is_satisfiable": is_satisfiable,
        "reachable_states": sorted(list(reachable_states)),
        "deadlock_states": deadlock_states,
    }
