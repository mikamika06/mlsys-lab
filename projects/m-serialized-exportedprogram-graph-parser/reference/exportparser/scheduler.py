def build_execution_schedule(graph_ir):
    """Generate a topologically sorted list of node names for execution."""
    inputs = set(graph_ir.get("inputs", []))
    nodes = graph_ir.get("nodes", [])

    ready = set(inputs)
    schedule = []
    remaining = list(nodes)

    while remaining:
        progress = False
        for i, node in enumerate(remaining):
            deps = set()
            for arg in node.get("args", []):
                if isinstance(arg, str) and arg.startswith("%"):
                    deps.add(arg[1:])
            for v in node.get("kwargs", {}).values():
                if isinstance(v, str) and v.startswith("%"):
                    deps.add(v[1:])

            if deps.issubset(ready):
                schedule.append(node["name"])
                ready.add(node["name"])
                remaining.pop(i)
                progress = True
                break

        if not progress:
            raise ValueError("Dependency cycle or missing input detected in graph.")

    return schedule


def verify_schedule(graph_ir, schedule):
    """Verify that every node appears after all its dependencies in the schedule."""
    node_names = [n["name"] for n in graph_ir.get("nodes", [])]
    if set(schedule) != set(node_names) or len(schedule) != len(node_names):
        return False

    position = {name: i for i, name in enumerate(schedule)}
    inputs = set(graph_ir.get("inputs", []))

    for node in graph_ir.get("nodes", []):
        name = node["name"]
        node_pos = position[name]

        for arg in node.get("args", []):
            if isinstance(arg, str) and arg.startswith("%"):
                dep = arg[1:]
                if dep not in inputs:
                    if dep not in position or position[dep] >= node_pos:
                        return False

        for v in node.get("kwargs", {}).values():
            if isinstance(v, str) and v.startswith("%"):
                dep = v[1:]
                if dep not in inputs:
                    if dep not in position or position[dep] >= node_pos:
                        return False

    return True
