def compute_region_times(tree):
    totals = {}

    def traverse(node):
        name = node.get("name")
        dur = float(node.get("duration", 0.0))
        children = node.get("children", [])
        child_dur = sum(float(c.get("duration", 0.0)) for c in children)
        self_dur = max(0.0, dur - child_dur)
        if name:
            totals[name] = totals.get(name, 0.0) + self_dur
        for c in children:
            traverse(c)

    traverse(tree)
    grand_total = sum(totals.values())
    if grand_total <= 0.0:
        return {k: 0.0 for k in totals}
    return {k: (v / grand_total) * 100.0 for k, v in totals.items()}
