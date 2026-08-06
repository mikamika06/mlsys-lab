def compare_checkpoints(checkpoints):
    points = [{"id": cp["id"], "params": cp["params"], "accuracy": cp["accuracy"]} for cp in checkpoints]
    sorted_pts = sorted(points, key=lambda x: (x["params"], -x["accuracy"]))
    front = []
    best_acc = -float("inf")
    for p in sorted_pts:
        if p["accuracy"] > best_acc:
            front.append(p)
            best_acc = p["accuracy"]
    front_ids = {p["id"] for p in front}
    results = []
    for cp in checkpoints:
        results.append({
            "id": cp["id"],
            "is_pareto": cp["id"] in front_ids,
            "params": cp["params"],
            "accuracy": cp["accuracy"]
        })
    return results
