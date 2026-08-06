import re


def extract_fusion_groups(code_str):
    pattern = r"triton_poi_fused_([a-zA-Z0-9_]+)"
    groups = re.findall(pattern, code_str)
    return sorted(list(set(groups)))
