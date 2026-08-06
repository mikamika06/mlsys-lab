from minjatools.detect import find_unsupported_filters

def needs_jinja_for_tools(template_str, tool_definitions):
    unsupported = find_unsupported_filters(template_str)
    has_tools_in_prompt = "tools" in template_str or len(tool_definitions) > 0
    if len(unsupported) > 0 and has_tools_in_prompt:
        return True
    if "raise" in template_str and len(tool_definitions) > 0:
        return True
    return False
