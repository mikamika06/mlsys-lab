def verify_index(index_dict, file_sizes_dict):
    weight_map = index_dict.get("weight_map", {})
    referenced_files = set(weight_map.values())
    actual_files = set(file_sizes_dict.keys())
    missing_files = sorted(list(referenced_files - actual_files))
    return {
        "is_valid": len(missing_files) == 0,
        "missing_files": missing_files,
        "total_files_referenced": len(referenced_files)
    }
