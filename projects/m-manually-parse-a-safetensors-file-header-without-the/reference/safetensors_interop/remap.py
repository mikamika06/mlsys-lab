import re


def match_and_remap_key(hf_key: str, rule_map: dict) -> str | None:
    for pattern, target_template in rule_map.items():
        if "{i}" in pattern:
            regex_pattern = (
                "^" + re.escape(pattern).replace(r"\{i\}", r"(?P<i>\d+)") + "$"
            )
            match = re.match(regex_pattern, hf_key)
            if match:
                return target_template.format(**match.groupdict())
        elif pattern == hf_key:
            return target_template
    return None


def remap_hf_to_mlx(hf_tensors: dict, rule_map: dict) -> tuple[dict, list[str]]:
    remapped = {}
    unmapped = []
    for key, val in hf_tensors.items():
        new_key = match_and_remap_key(key, rule_map)
        if new_key is not None:
            remapped[new_key] = val
        else:
            unmapped.append(key)
    return remapped, unmapped
