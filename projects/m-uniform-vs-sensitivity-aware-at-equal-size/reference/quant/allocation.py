def allocate_uniform(layers, target_size_bytes, bits_options):
    alloc = {}
    for l in layers:
        if l["sensitivity"] == 0.0:
            alloc[l["name"]] = 8
            continue
        alloc[l["name"]] = 4
    return alloc


def allocate_sensitivity(layers, target_size_bytes, bits_options):
    alloc = {}
    sorted_layers = sorted([l for l in layers if l["sensitivity"] > 0], key=lambda x: x["sensitivity"], reverse=True)
    for i, l in enumerate(sorted_layers):
        if i < 2:
            alloc[l["name"]] = 8
        else:
            alloc[l["name"]] = 2
    for l in layers:
        if l["sensitivity"] == 0.0:
            alloc[l["name"]] = 8
    return alloc
