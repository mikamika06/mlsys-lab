import ref


def check(workdir):
    from dataloader.parser import parse_nvtx_timeline

    cases = ref.get_test_cases()
    matched = 0
    for events, _ in cases:
        want = []
        stack = []
        for ts, kind, name in sorted(events, key=lambda x: x[0]):
            if kind == "push":
                stack.append((ts, name))
                want.append((ts, len(stack), name))
            elif kind == "pop":
                if stack:
                    _, popped_name = stack.pop()
                    want.append((ts, len(stack) + 1, popped_name))

        got = parse_nvtx_timeline(events)
        if got == want:
            matched += 1

    return {"depths_matched": float(matched)}
