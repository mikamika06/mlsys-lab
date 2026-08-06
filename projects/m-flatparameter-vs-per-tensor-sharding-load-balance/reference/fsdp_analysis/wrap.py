def assign_units(modules, min_num_params):
    units = []
    current_unit = []
    current_size = 0
    for m in modules:
        current_unit.append(m["name"])
        current_size += m["size"]
        if current_size >= min_num_params:
            units.append(current_unit)
            current_unit = []
            current_size = 0
    if current_unit:
        if units:
            units[-1].extend(current_unit)
        else:
            units.append(current_unit)
    return units
