def extract_initializers(model_dict: dict) -> list:
    manifest = []
    for g in model_dict.get("graphs", []):
        for init in g.get("initializers", []):
            manifest.append({
                "name": init.get("name"),
                "dims": init.get("dims"),
                "data_type": init.get("data_type"),
                "size_bytes": len(init.get("raw_data", b""))
            })
    return manifest
