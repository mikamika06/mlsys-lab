import json


def measure_quality_delta(reference_template, candidate_template, dataset):
    total = len(dataset)
    if total == 0:
        return {
            "exact_match_rate": 0.0,
            "system_retention_rate": 0.0,
            "tool_schema_error_rate": 0.0,
            "quality_delta": 0.0,
        }

    exact_matches = 0
    system_retained = 0
    tool_schema_errors = 0

    for item in dataset:
        ref_rendered = reference_template.render(
            system=item.get("system"),
            messages=item.get("messages", []),
            tools=item.get("tools"),
        )
        cand_rendered = candidate_template.render(
            system=item.get("system"),
            messages=item.get("messages", []),
            tools=item.get("tools"),
        )

        if ref_rendered == cand_rendered:
            exact_matches += 1

        sys_prompt = item.get("system", "")
        if sys_prompt:
            if sys_prompt in cand_rendered:
                system_retained += 1
        else:
            system_retained += 1

        tools = item.get("tools")
        if tools:
            has_tool_error = False
            for t in tools:
                t_name = t.get("name")
                if t_name and f'"name": "{t_name}"' not in cand_rendered:
                    has_tool_error = True
                    break
            if has_tool_error:
                tool_schema_errors += 1

    em_rate = exact_matches / total
    sys_rate = system_retained / total
    tool_err_rate = tool_schema_errors / total

    quality_delta = (1.0 - em_rate) * 0.4 + (1.0 - sys_rate) * 0.3 + tool_err_rate * 0.3

    return {
        "exact_match_rate": em_rate,
        "system_retention_rate": sys_rate,
        "tool_schema_error_rate": tool_err_rate,
        "quality_delta": quality_delta,
    }
