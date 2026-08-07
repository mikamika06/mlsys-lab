def dump_initializers(graph_dict: dict) -> list:
    manifests = []
    for init in graph_dict.get("initializers", []):
        name = init.get("name", "")
        dims = init.get("dims", [])
        data_type = init.get("data_type", 1)
        raw_data = init.get("raw_data", b"")
        size_bytes = len(raw_data)
        manifests.append({
            "name": name,
            "dims": dims,
            "data_type": data_type,
            "size_bytes": size_bytes
        })
    return manifests
