def count_nodes(schema):
    if isinstance(schema, dict):
        return 1 + sum(count_nodes(v) for v in schema.values())
    elif isinstance(schema, list):
        return sum(count_nodes(item) for item in schema)
    return 0
