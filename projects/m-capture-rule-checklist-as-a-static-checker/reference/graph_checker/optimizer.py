import copy
import torch
import torch.fx

def suggest_safe_transforms(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Rewrite simple dynamic allocation ops into static placeholder inputs."""
    new_gm = copy.deepcopy(gm)
    graph = new_gm.graph
    nodes_to_remove = []
    
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in (torch.empty, torch.zeros, torch.ones):
            with graph.inserting_before(node):
                zero_const = graph.call_function(torch.zeros_like, args=(node.args[0],)) if node.args else None
                if zero_const is not None:
                    node.replace_all_uses_with(zero_const)
                    nodes_to_remove.append(node)
        elif node.op == "call_method" and node.target in ("cpu", "cuda"):
            if len(node.args) > 0:
                node.replace_all_uses_with(node.args[0])
                nodes_to_remove.append(node)

    for node in nodes_to_remove:
        graph.erase_node(node)
        
    graph.lint()
    new_gm.recompile()
    return new_gm
