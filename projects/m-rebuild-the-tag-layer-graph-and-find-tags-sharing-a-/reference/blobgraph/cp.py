def simulate_cp(tags_dict, src_tag, dst_tag):
    if src_tag not in tags_dict:
        raise KeyError(f"Source tag {src_tag} not found")
    return 0
