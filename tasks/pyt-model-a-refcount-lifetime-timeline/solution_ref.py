def refcount_timeline(steps):
    count = 1
    timeline = []

    for step in steps:
        if step == "assign":
            count += 1
        elif step == "alias":
            count += 1
        elif step == "container-insert":
            count += 1
        elif step == "function-arg":
            pass
        elif step == "return":
            count += 1
        else:
            raise ValueError(step)
        timeline.append(count)

    return timeline
