def handle_tools(tools_spec):
    if not tools_spec:
        return []
    formatted = []
    for t in tools_spec:
        formatted.append({
            "type": "function",
            "function": t
        })
    return formatted
