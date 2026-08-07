def find_pareto_front(results):
    points = [(r["ttft"], r["itl"], r) for r in results]
    points.sort(key=lambda x: x[0])
    pareto = []
    min_itl = float("inf")
    for ttft, itl, r in points:
        if itl < min_itl:
            pareto.append(r)
            min_itl = itl
    return pareto
