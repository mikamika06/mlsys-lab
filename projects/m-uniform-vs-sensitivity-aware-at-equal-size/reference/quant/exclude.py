def apply_exclusions(layers, excluded_names):
    out = []
    for l in layers:
        if l["name"] in excluded_names:
            new_l = l.copy()
            new_l["excluded"] = True
            out.append(new_l)
        else:
            new_l = l.copy()
            new_l["excluded"] = False
            out.append(new_l)
    return out
