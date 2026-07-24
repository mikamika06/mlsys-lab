import random
import collections

# Policy parameters – must match reference implementation
WINDOW_SIZE = 5
HEAVY_HITTER_K = 3
RECENT_ONLY_N = 5


def _window_policy(stream):
    cache = collections.OrderedDict()
    for t in stream:
        if t in cache:
            del cache[t]
        cache[t] = None
        if len(cache) > WINDOW_SIZE:
            cache.popitem(last=False)
    return list(reversed(list(cache.keys())))


def _heavy_hitter_policy(stream):
    freq = collections.Counter(stream)
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in items[:HEAVY_HITTER_K]]


def _recent_only_policy(stream):
    dq = collections.deque(maxlen=RECENT_ONLY_N)
    for t in stream:
        dq.append(t)
    return list(dq)


def grade(sol, fx) -> dict:
    """
    For each test case we generate a random token stream,
    compute the retained set from each policy, pick one at random
    and ask the candidate solution to identify it.
    """
    ok = 1.0
    for _ in range(20):
        # Ensure distinct outputs by regenerating if needed
        while True:
            length = random.randint(10, 50)
            stream = [random.randint(0, 9) for _ in range(length)]
            w_set = _window_policy(stream)
            h_set = _heavy_hitter_policy(stream)
            r_set = _recent_only_policy(stream)
            if len({tuple(w_set), tuple(h_set), tuple(r_set)}) == 3:
                break

        # Pick one policy at random
        choice = random.choice(["window", "heavy_hitter", "recent_only"])
        if choice == "window":
            target_set, expected_label = w_set, "window"
        elif choice == "heavy_hitter":
            target_set, expected_label = h_set, "heavy_hitter"
        else:
            target_set, expected_label = r_set, "recent_only"

        try:
            got = sol.identify_policy(target_set, stream)
        except Exception:
            ok = 0.0
            break

        if got != expected_label:
            ok = 0.0
            break

    return {"exact_match": ok}
