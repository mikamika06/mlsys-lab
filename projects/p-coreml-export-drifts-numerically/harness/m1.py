def check(workdir):
    from coreml_export.comparator import LayerComparator
    m = {"layers_compared": 0.0, "max_diff_tracked": 0.0}
    comp = LayerComparator()
    layers = comp.compare_layers()
    if isinstance(layers, dict) and len(layers) >= 3:
        m["layers_compared"] = float(len(layers))
    if comp.has_max_diff():
        m["max_diff_tracked"] = 1.0
    return m
