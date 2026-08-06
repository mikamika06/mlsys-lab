def label_joint_graph(graph):
    labeled = []
    for node in graph["nodes"]:
        n = dict(node)
        n["phase"] = "forward" if node.get("is_fw_target", False) else "backward"
        labeled.append(n)
    return labeled
