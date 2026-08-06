import os


def find_initializer_inputs(model_proto):
    init_names = {init.name for init in model_proto.graph.initializer}
    input_names = {inp.name for inp in model_proto.graph.input}
    return sorted(list(init_names.intersection(input_names)))


def resolve_external_ranges(model_proto, base_dir):
    results = {}
    for init in model_proto.graph.initializer:
        ext_entries = {}
        for entry in init.external_data:
            ext_entries[entry.key] = entry.value
        if ext_entries:
            loc = ext_entries.get("location", "")
            offset = int(ext_entries.get("offset", "0"))
            length = int(ext_entries.get("length", "0"))
            full_path = os.path.normpath(os.path.join(base_dir, loc))
            if length == 0 and os.path.exists(full_path):
                length = os.path.getsize(full_path) - offset
            results[init.name] = {"path": full_path, "offset": offset, "length": length}
    return results


def predict_2gb_ceiling(model_proto):
    total_size = 0
    for init in model_proto.graph.initializer:
        if not init.external_data:
            if init.raw_data:
                total_size += len(init.raw_data)
            else:
                sz = 4
                for d in init.dims:
                    sz *= d
                total_size += sz
    return total_size > 2147483648
