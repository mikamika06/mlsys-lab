def select_pareto_front(points):
    sorted_pts = sorted(points, key=lambda x: (x["params"], -x["accuracy"]))
    front = []
    best_acc = -float("inf")
    for p in sorted_pts:
        if p["accuracy"] > best_acc:
            front.append(p)
            best_acc = p["accuracy"]
    return front
