import copy

def fuse_conv_bn(graph):
    nodes = graph["nodes"]
    new_nodes = []
    i = 0
    while i < len(nodes):
        if i + 1 < len(nodes) and nodes[i]["op"] == "torch.conv2d" and nodes[i+1]["op"] == "torch.batch_norm":
            merged = copy.deepcopy(nodes[i])
            merged["name"] = "conv_bn_fused"
            merged["op"] = "torch.conv_bn"
            merged["output"] = nodes[i+1]["output"]
            new_nodes.append(merged)
            i += 2
        else:
            new_nodes.append(copy.deepcopy(nodes[i]))
            i += 1
    delta = len(new_nodes) - len(nodes)
    return {"nodes": new_nodes}, delta
