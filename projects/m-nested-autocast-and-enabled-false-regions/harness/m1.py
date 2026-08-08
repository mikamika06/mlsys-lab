import ref


def check(workdir):
    from autocast.engine import simulate_stack
    out = {"states_matched": 0.0}
    ok = 0
    for events in ref.EVENTS:
        want = []
        stack = []
        for ev in events:
            if ev["type"] == "push":
                stack.append({"device": ev["device"], "dtype": ev["dtype"], "enabled": ev["enabled"]})
            elif ev["type"] == "pop":
                if stack:
                    stack.pop()
            want.append(dict(stack[-1]) if stack else {"device": None, "dtype": None, "enabled": False})
        try:
            got = simulate_stack(events)
            if got == want:
                ok += 1
        except Exception:
            pass
    out["states_matched"] = float(ok)
    return out
