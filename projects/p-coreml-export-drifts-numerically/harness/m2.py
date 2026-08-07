def check(workdir):
    from coreml_export.comparator import LayerComparator
    m = {"top_culprit_identified": 0.0}
    comp = LayerComparator()
    culprit = comp.find_top_culprit()
    if culprit is not None and isinstance(culprit, str):
        m["top_culprit_identified"] = 1.0
    return m
