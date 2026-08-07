def parse_dcp_metadata(metadata_dict):
    storage_data = metadata_dict.get("storage_data", {})
    plans = {}
    for param_name, info in storage_data.items():
        plans[param_name] = {
            "shape": info.get("shape"),
            "offsets": info.get("offsets", []),
            "lengths": info.get("lengths", []),
            "file_name": info.get("file_name")
        }
    return plans
