def flatten_exported_inputs(tree, input_spec):
    def get_path(value, path):
        current = value
        for part in path:
            current = current[part]
        return current

    return [get_path(tree, entry["path"]) for entry in input_spec]
